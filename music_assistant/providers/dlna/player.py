"""DLNA/uPNP Player provider for Music Assistant.

Most of this code is based on the implementation within Home Assistant:
https://github.com/home-assistant/core/blob/dev/homeassistant/components/dlna_dmr

All rights/credits reserved.
"""

from __future__ import annotations

from asyncio import Lock, Task, create_task, sleep
from collections.abc import Awaitable, Callable, Coroutine, Mapping  # pylint: disable=import-error
from functools import wraps
from time import time
from typing import TYPE_CHECKING, Any, ParamSpec

from async_upnp_client.exceptions import UpnpError, UpnpResponseError
from async_upnp_client.profiles.dlna import DmrDevice, TransportState, split_commas
from music_assistant_models.enums import (
    MediaType,
    PlayerFeature,
    PlayerState,
    PlayerType,
)
from music_assistant_models.errors import PlayerUnavailableError
from music_assistant_models.player import DeviceInfo, Player, PlayerSource

from .ssdp_listener import ATTR_SSDP_BOOTID, ATTR_SSDP_CONFIGID, ATTR_SSDP_LOCATION

if TYPE_CHECKING:
    from collections.abc import KwArg, Sequence, VarArg, _Wrapped

    from async_upnp_client.client import (  # type: ignore[attr-defined]
        UpnpEventHandler,
        UpnpService,
        UpnpStateVariable,
    )
    from async_upnp_client.client_factory import UpnpFactory
    from async_upnp_client.utils import CaseInsensitiveDict

    from .provider import DLNAPlayerProvider


# Map UPnP class to media_player media_content_type
_MEDIA_TYPE_MAP: Mapping[str, MediaType] = {
    "object": MediaType.UNKNOWN,
    "object.item": MediaType.UNKNOWN,
    "object.item.audioItem": MediaType.TRACK,
    "object.item.audioItem.musicTrack": MediaType.TRACK,
    "object.item.audioItem.audioBroadcast": MediaType.FLOW_STREAM,
    "object.item.audioItem.audioBook": MediaType.AUDIOBOOK,
    "object.item.playlistItem": MediaType.PLAYLIST,
    "object.container": MediaType.PLAYLIST,
    "object.container.person": MediaType.ARTIST,
    "object.container.person.musicArtist": MediaType.ARTIST,
    "object.container.playlistContainer": MediaType.PLAYLIST,
    "object.container.album": MediaType.ALBUM,
    "object.container.album.musicAlbum": MediaType.ALBUM,
    "object.container.album.photoAlbum": MediaType.ALBUM,
    "object.container.genre": MediaType.PLAYLIST,
    "object.container.genre.musicGenre": MediaType.PLAYLIST,
    "object.container.storageSystem": MediaType.PLAYLIST,
    "object.container.storageVolume": MediaType.PLAYLIST,
    "object.container.storageFolder": MediaType.PLAYLIST,
    "object.container.bookmarkFolder": MediaType.PLAYLIST,
}

# Map UPnP TransportState to mass PlayerState
_TRANSPORT_STATE_TO_PLAYER_STATE: Mapping[TransportState, PlayerState] = {
    TransportState.STOPPED: PlayerState.IDLE,
    TransportState.NO_MEDIA_PRESENT: PlayerState.IDLE,
    TransportState.PLAYING: PlayerState.PLAYING,
    TransportState.TRANSITIONING: PlayerState.PLAYING,
    TransportState.PAUSED_PLAYBACK: PlayerState.PAUSED,
    TransportState.PAUSED_RECORDING: PlayerState.PAUSED,
    # Unable to map this state to anything reasonable, fallback to idle
    TransportState.VENDOR_DEFINED: PlayerState.IDLE,
    TransportState.RECORDING: PlayerState.IDLE,
}


P = ParamSpec("P")  # the callable parameters


def debounce(wait: float) -> Callable[[Callable[P, Awaitable[Any]]], Callable[P, Awaitable[Any]]]:
    """Decorato a func to debounce and throttle.

    Debounce and throttle an async function for a given amount of time in seconds.

    !!!Be aware that this wrapper might skip calls to the function if they occur too quickly!!!
    """

    def decorator(
        func: Callable[P, Any],
    ) -> _Wrapped[P, Any, [VarArg(Any), KwArg(Any)], Coroutine[Any, Any, None]]:
        task: Task[Any] | None = None
        last_run: float | None = None

        @wraps(func)
        async def throttle(*args: Any, **kwargs: Any) -> Task[Any] | None:
            nonlocal task, last_run

            # if a task already exists, cancel it
            if task and not task.done():
                return None

            if last_run and (time() - last_run) > wait:
                return None

            # create a task which will run in {wait} time.
            #
            # needs to be saved to a var instead of being returned directly.
            # otherwise the task is executed right away.
            task = create_task(func(*args, **kwargs))

            return task

        return throttle

    return decorator


class DLNAPlayer:
    """Class that holds all dlna variables for a player."""

    # DLNA/DMR device
    dmr_device: DmrDevice | None = None

    # Signals if this device supports FLAC.
    # We proactively assume it does because because we can only
    # check it correctly at runtime.
    # At least as "correctly" as those many broken DLNA-clients out there are…
    supports_flac: bool = True

    # Music Assistant Player instance
    _player: Player

    # Held when connecting or disconnecting the device
    _lock: Lock

    def __init__(
        self,
        provider: DLNAPlayerProvider,
        udn: str,
        upnp_factory: UpnpFactory,
        event_handler: UpnpEventHandler,
    ) -> None:
        """Initialize the DLNA player.

        :param provider: The owning Player provider.
        :param udn: Unique Device Name, the name that uniquely identifies the specific device.
        :param upnp_factory: UpnpFactory to instantiate a UpnpDevice from a description URL.
        :param event_handler: The EventHandler for DLNA notifications.
        """
        self.provider = provider
        self._player = Player(
            player_id=udn,
            provider=self.provider.instance_id,
            type=PlayerType.PLAYER,
            name=udn,
            available=False,
            powered=False,
            device_info=DeviceInfo(
                model="unknown",
                manufacturer="unknown",
            ),
        )
        self.logger = self.provider.logger.getChild(self._player.player_id)
        self._upnp_factory = upnp_factory
        self._event_handler = event_handler
        self._lock = Lock()

    @property
    def id(self) -> str:
        """Player id."""
        return self._player.player_id

    @property
    def name(self) -> str:
        """Player name."""
        return self._player.name

    @property
    def connected(self) -> bool:
        """Player connection status.

        Returns True if a DMR Device exists.

        Check self.available to see if the device is available.
        """
        return self.dmr_device is not None

    @property
    def available(self) -> bool:
        """Player availability.

        Device is available when DLNA device is there and available.
        """
        return (
            self.connected
            and self.dmr_device is not None
            and self.dmr_device.profile_device.available
        )

    def get_device(self) -> DmrDevice:
        """Get the Digital Media Renderer device."""
        if not self.available or self.dmr_device is None:
            raise PlayerUnavailableError

        return self.dmr_device

    async def async_connect(
        self, location: str, headers: CaseInsensitiveDict | None = None
    ) -> None:
        """Connect to the DLNA/DMR Device.

        Can always be called and will try to ensure a recent connection to the device.
        We handle necessary reconnects and reinitialization internally.

        :param location: The description URL. Allowed to change within the lifetime of the device.
        :param headers: Optional combined headers retrieved via SSDP to determine state changes.
        """
        async with self._lock:
            try:
                self.logger.debug("Connecting to device at %s", location)

                do_reinit = False

                # handle first connect
                if self.dmr_device is None:
                    self.logger.debug("First connect, creating dmr device")

                    # Connect to the base UPNP device
                    self.dmr_device = await self._async_create_device(location)
                elif headers:
                    # Handle BOOTID.UPNP.ORG.
                    boot_id = headers.get(ATTR_SSDP_BOOTID)
                    device_boot_id = self.dmr_device.profile_device.ssdp_headers.get(
                        ATTR_SSDP_BOOTID
                    )
                    if boot_id and boot_id != device_boot_id:
                        self.logger.info(
                            "Found changed boot_id: %s, old boot_id: %s",
                            boot_id,
                            device_boot_id,
                        )
                        do_reinit = True

                    # Handle CONFIGID.UPNP.ORG.
                    config_id = headers.get(ATTR_SSDP_CONFIGID)
                    device_config_id = self.dmr_device.profile_device.ssdp_headers.get(
                        ATTR_SSDP_CONFIGID
                    )
                    if config_id and config_id != device_config_id:
                        self.logger.info(
                            "Found changed config_id: %s, old config_id: %s",
                            config_id,
                            device_config_id,
                        )
                        do_reinit = True

                    if ATTR_SSDP_LOCATION in headers:
                        location = str(headers.get(ATTR_SSDP_LOCATION))

                device_location = self.dmr_device.profile_device.device_url
                if location != device_location:
                    self.logger.info(
                        "found changed location: %s, old location: %s",
                        location,
                        device_location,
                    )
                    do_reinit = True

                if do_reinit:
                    if headers:
                        await self._async_reinit_device(location, headers)
                    else:
                        # this is only needed when the connect is triggered manually
                        # without providing headers
                        self.dmr_device = await self._async_create_device(location)

                # Subscribe to event notifications
                self.dmr_device.on_event = self._handle_upnp_event

                # connect was successful, update device info
                self._player.device_info = DeviceInfo(
                    ip_address=self.dmr_device.profile_device.device_url or location,
                    manufacturer=self.dmr_device.manufacturer,
                    model=self.dmr_device.model_name,
                    model_id=self.dmr_device.model_number.strip()
                    if self.dmr_device.model_number
                    else None,
                )
                self._player.powered = True
                self.dmr_device.profile_device.available = True

                # subscribe to DLNA-events. Automatically resubscribes if already subscribed.
                await self.dmr_device.async_subscribe_services(auto_resubscribe=True)

                self._update_supported_features()

                await self.provider.mass.players.register_or_update(self._player)

            except UpnpResponseError as err:
                # Device rejected subscription request. This is OK, variables
                # will be polled instead.
                self.logger.debug("Device rejected subscription: %r", err)

                self._player.needs_poll = True
            except UpnpError as err:
                # Don't leave the device half-constructed
                if self.dmr_device:
                    self.dmr_device.on_event = None
                self.dmr_device = None
                self.logger.debug("Error while subscribing during device connect: %r", err)
                raise

    async def async_disconnect(self) -> None:
        """Disconnect the device.

        Call this method when this entity will be removed from MA to clean up connections.

        Destroy all connections to the device.
        """
        async with self._lock:
            self._player.available = False

            if not self.dmr_device:
                self.logger.debug("Disconnecting from device that's not connected")
                return

            self.logger.debug("Disconnecting from %s", self.dmr_device.name)

            self.dmr_device.on_event = None
            old_device = self.dmr_device
            self.dmr_device = None
            await old_device.async_unsubscribe_services()

            self.provider.mass.players.update(self.id)

    @debounce(1)
    async def async_update(self, do_ping: bool = False) -> None:
        """Retrieve the latest data from the player.

        :param do_ping: Poll device to check if it is available (online).

        This will be called after retrieving events from the DLNA device.
        Those events can occur in quick succession and therefore this method
        is debounced and will run once every second at most.


        """
        assert self.dmr_device is not None

        self.logger.info("polling player")

        try:
            # Poll the player for unevented vars
            await self.dmr_device.async_update(do_ping=do_ping)
        except UpnpError as err:
            self.logger.debug("Device unavailable: %r", err)
            await self.async_disconnect()
            return

        # Wait for events to arrive
        await sleep(1)

        prev_url = self._player.current_item_id
        prev_state = self._player.state
        self._update_attributes()
        current_url = self._player.current_item_id
        current_state = self._player.state

        if (prev_url != current_url) or (prev_state != current_state):
            # fetch track details on state or url change
            self.logger.info("media item changed…")

        self.logger.debug("write player state")

        # inform MA off changes and write the player state
        self.provider.mass.players.update(self.id)

    def _handle_upnp_event(
        self,
        service: UpnpService,
        state_variables: Sequence[UpnpStateVariable[Any]],
    ) -> None:
        """Handle state variable(s) changed event from DLNA device.

        This method also triggers a debounced update to the player state.

        :param service: The UpnpService that produced the event.
        :param state_variables: Variables that changed within the event.
        """
        if not state_variables:
            # TODO handle subscription-failures properly
            # Indicates a failure to resubscribe, check if device is still available
            # self.check_available = True
            return

        assert self.dmr_device is not None

        if service.service_id == "urn:upnp-org:serviceId:AVTransport":
            for state_variable in state_variables:
                if state_variable.name == "SinkProtocolInfo":
                    # Does this device support flac?
                    self.supports_flac = False

                    for item in split_commas(state_variable.value):
                        if "audio/flac" in item.lower():
                            self.supports_flac = True
                            self.logger.debug("Player supports flac")
                            break
                    else:
                        self.logger.debug("Player does not support flac")

                elif state_variable.name == "PlaybackStorageMedium":
                    active_source = None

                    if self._is_mass_stream():
                        active_source = self.id

                    elif state_variable.value:
                        self._player.source_list.clear()

                        active_source = state_variable.value.lower()

                        if active_source in ("unknown", "none", "un_known", ""):
                            active_source = None

                        if active_source:
                            self._player.source_list.append(
                                PlayerSource(
                                    id=active_source.lower(),
                                    name=active_source.title(),
                                    passive=True,
                                    can_play_pause=self.dmr_device.can_pause
                                    and self.dmr_device.can_play,
                                    can_next_previous=self.dmr_device.can_pause
                                    and self.dmr_device.can_play,
                                    can_seek=self.dmr_device.can_seek_abs_time,
                                )
                            )

                    self._player.active_source = active_source

                if state_variable.name == "TransportState":
                    self.logger.debug(
                        "Received new transport state for Player %s: %s",
                        self.name,
                        state_variable.value,
                    )

                    if state_variable.value == TransportState.PLAYING:
                        # Because not every action produces events per DLNA spec,
                        # additionally we poll for changes every few seconds.
                        self._player.needs_poll = True
                        self._player.poll_interval = 3

                        self.logger.debug("Start timed polling.")
                    else:
                        self._player.needs_poll = False
                        self._player.poll_interval = 30

                        self.logger.debug("Stop timed polling.")

                        if self.dmr_device.transport_state in (
                            TransportState.STOPPED,
                            TransportState.TRANSITIONING,
                        ):
                            self._player.elapsed_time = None
                            self._player.elapsed_time_last_updated = None

        self.provider.mass.create_task(self.async_update())

    def _update_attributes(self) -> None:
        """Update attributes of the MA Player from DLNA state."""
        assert self.dmr_device is not None

        # generic attributes
        if self.available:
            self._player.available = True
            self._player.name = self.dmr_device.name
            self._player.volume_level = int((self.dmr_device.volume_level or 0) * 100)
            self._player.volume_muted = self.dmr_device.is_volume_muted or False
            self._player.state = _TRANSPORT_STATE_TO_PLAYER_STATE[
                self.dmr_device.transport_state or TransportState.STOPPED
            ]
            self._player.current_item_id = self.dmr_device.current_track_uri or ""

            # set active source

            if self.id in self._player.current_item_id:
                self._player.active_source = self.id
            else:
                # sync the received metadata of the currently played media to MA.
                self._player.set_current_media(
                    uri=self.dmr_device.current_track_uri or "",
                    media_type=_MEDIA_TYPE_MAP.get(self.dmr_device.media_class or "object")
                    or MediaType.UNKNOWN,
                    title=self.dmr_device.media_title,
                    artist=self.dmr_device.media_artist,
                    album=self.dmr_device.media_album_name,
                    image_url=self.dmr_device.media_image_url,
                    duration=self.dmr_device.media_duration,
                )

            self._player.elapsed_time = (
                float(self.dmr_device.media_position)
                if self.dmr_device.media_position is not None
                else None
            )

            # we need to convert the utc-datetime DLNA uses to the
            # local timezone MA is expecting. After that, we convert it to
            self._player.elapsed_time_last_updated = (
                (self.dmr_device.media_position_updated_at.timestamp())
                if self.dmr_device.media_position_updated_at is not None
                else None
            )

            self.logger.warning(
                f"DLNA {self.dmr_device.media_position} {
                    self.dmr_device.media_position_updated_at.timestamp()
                    if self.dmr_device.media_position_updated_at is not None
                    else None
                }"
            )

            self.logger.warning(
                f"MASS {self._player.elapsed_time} {self._player.elapsed_time_last_updated}"
            )

            self._update_supported_features()
        else:
            # device is unavailable
            self._player.available = False

    def _update_supported_features(self) -> None:
        """Set Player Features based on config values and capabilities."""
        if not self.dmr_device:
            return

        # Update supported features
        supported_features = set[PlayerFeature]()

        if self.dmr_device.has_next_transport_uri:
            supported_features.add(PlayerFeature.ENQUEUE)
        if self.dmr_device.has_volume_level:
            supported_features.add(PlayerFeature.VOLUME_SET)
        if self.dmr_device.has_volume_mute:
            supported_features.add(PlayerFeature.VOLUME_MUTE)
        if self.dmr_device.has_pause:
            supported_features.add(PlayerFeature.PAUSE)
        if self.dmr_device.has_seek_abs_time:
            supported_features.add(PlayerFeature.SEEK)
        if self.dmr_device.has_next and self.dmr_device.has_previous:
            supported_features.add(PlayerFeature.NEXT_PREVIOUS)

        self._player.supported_features = supported_features

    async def _async_create_device(self, location: str) -> DmrDevice:
        """Create the device.

        Create the DMR device. Before that, we ensure the device provides the needed
        services for the DMR profile. If those are not found, an error is raised and the player
        is deleted by the provider.
        """
        # Connect to the base UPNP device
        upnp_device = await self._upnp_factory.async_create_device(location)

        # Check if the device is a DMR-profile device.
        # We are doing this here and this late to avoid fetching and parsing the
        # description.xml twice.
        if not DmrDevice.is_profile_device(upnp_device):
            self.provider.mass.players.remove(self.id)

            # will be caught by the provider which will delete the device.
            raise RuntimeError(f"Found incompatible device, removing: {upnp_device.device_type}")

        # Create profile wrapper
        return DmrDevice(upnp_device, self._event_handler)

    async def _async_reinit_device(self, location: str, ssdp_headers: CaseInsensitiveDict) -> None:
        """Reinitialize the device.

        Reinitialize on reboots and config changes. This is signaled via SSDP and needed because,
        technically, devices can change provided services and subdevices while online.
        """
        assert self.dmr_device is not None

        self.logger.debug("Reinitializing device, location: %s", location)

        new_device = await self._upnp_factory.async_create_device(location)
        self.dmr_device.profile_device.reinit(new_device)
        self.dmr_device.profile_device.ssdp_headers = ssdp_headers

    def _is_mass_stream(self) -> bool:
        return bool(self._player.current_item_id and self.id in self._player.current_item_id)
