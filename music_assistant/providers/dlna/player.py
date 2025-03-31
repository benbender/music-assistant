"""DLNA/uPNP Player provider for Music Assistant.

Most of this code is based on the implementation within Home Assistant:
https://github.com/home-assistant/core/blob/dev/homeassistant/components/dlna_dmr

All rights/credits reserved.
"""

from __future__ import annotations

from asyncio import Lock, Task, sleep
from collections.abc import Mapping  # pylint: disable=import-error
from time import time
from typing import TYPE_CHECKING, Any

from async_upnp_client.exceptions import UpnpError, UpnpResponseError
from async_upnp_client.profiles.dlna import (
    DmrDevice,
    TransportState,
    _lower_split_commas,
)
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
    from collections.abc import Sequence

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


class DLNAPlayer:
    """Class that holds all dlna variables for a player."""

    # DLNA/DMR device
    dmr_device: DmrDevice | None = None

    # Signals if this device supports FLAC.
    # The actual support will be derived by the devices capabilities on runtime.
    supports_flac: bool = False

    # Music Assistant Player instance
    _player: Player

    # Held when connecting or disconnecting the device
    _lock: Lock

    # Update media task
    _last_media_update: float | None = None
    _update_media_task: Task[Any] | None = None

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

        Player is available when the DLNA device is created and available.
        """
        self._player.available = (
            self.connected
            and self.dmr_device is not None
            and self.dmr_device.profile_device.available
        )

        return self._player.available

    @property
    def queue_playing(self) -> bool:
        """Is playing a MA Queue."""
        return bool(self._player.current_item_id and self.id in self._player.current_item_id)

    def get_device(self) -> DmrDevice:
        """Get the Digital Media Renderer device."""
        if not self.available or self.dmr_device is None:
            raise PlayerUnavailableError

        return self.dmr_device

    async def async_connect(
        self, location: str, headers: CaseInsensitiveDict | None = None
    ) -> None:
        """Connect to the DLNA/DMR Device.

        Can always be called and will try to ensure a recent connection to the Device.
        We handle necessary reconnects and reinitialization internally.

        :param location: The description URL. Allowed to change within the lifetime of the device.
        :param headers: Optional combined headers retrieved via SSDP to determine state changes.
        """
        async with self._lock:
            try:
                self.logger.info(f"Connect to Player at {location}")

                do_reinit = False

                # handle first connect
                if self.dmr_device is None:
                    self.logger.debug("First connect, create DMR Device")

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
                    await self._async_reinit_device(location, headers)

                self._player.powered = True

                # Subscribe to event notifications
                self.dmr_device.on_event = self._handle_upnp_event

                # subscribe to DLNA-events. Automatically resubscribes if already subscribed.
                await self.dmr_device.async_subscribe_services(auto_resubscribe=True)

                # needs to be called before registering the player to get the correct
                # features set
                await self.update_player()

                await self.provider.mass.players.register_or_update(self._player)

                self.update_media()

            except UpnpResponseError as err:
                # Device rejected subscription request. This is OK, variables
                # will be polled instead.
                # TODO handle correctly
                self.logger.debug("Device rejected subscription: %r", err)

                # self._player.needs_poll = True
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

            if self._update_media_task and not self._update_media_task.done():
                self._update_media_task.cancel()

            if not self.dmr_device:
                self.logger.debug("Disconnect from device that's not connected")
                return

            self.logger.debug("Disconnect from %s", self.dmr_device.name)

            self.dmr_device.on_event = None
            old_device = self.dmr_device
            self.dmr_device = None
            await old_device.async_unsubscribe_services()

            self.provider.mass.players.update(self.id)

    async def update_player(self) -> None:
        """Update the Player features and capabilities."""
        if not self.dmr_device:
            return

        # Update basic info
        self._player.available = self.dmr_device.profile_device.available
        self._player.name = self.dmr_device.name
        self._player.device_info = DeviceInfo(
            ip_address=self.dmr_device.profile_device.device_url,
            manufacturer=self.dmr_device.manufacturer,
            model=self.dmr_device.model_name,
            model_id=self.dmr_device.model_number.strip() if self.dmr_device.model_number else None,
        )

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

        # Determine if the Device supports FLAC
        protocol_info = await self.dmr_device.async_get_protocol_info()

        for item in protocol_info["sink"]:
            if "audio/flac" in item:
                self.supports_flac = True
                self.logger.debug(f"Player {self.name} supports flac")
                break
        else:
            self.logger.debug(f"Player {self.name} does not support flac")

        # Dynamically create sources provided by the player
        if self._possible_sources:
            self._player.source_list.clear()

            for source in self._possible_sources:
                if source in ("none", "unknown", "un_known", ""):
                    continue

                self._player.source_list.append(
                    PlayerSource(
                        id=source.lower(),
                        name=source.title(),
                        passive=True,
                        can_play_pause=False,
                        can_next_previous=False,
                        can_seek=False,
                    )
                )

    def update_media(self, do_ping: bool = False) -> None:
        """Create a queued task to poll the latest data from the Player.

        :param do_ping: Poll device to check if it is available (online).

        This will be called after retrieving events from the DLNA device.
        Those events can occur in quick succession and therefore this method
        is debounced and throttled. It will run once per second at most.
        """
        # If a task already exists, skip
        if self._update_media_task and not self._update_media_task.done():
            return

        # If a last task ran less than a second ago, skip
        if self._last_media_update and (time() - self._last_media_update) < 1:
            return

        self._update_media_task = self.provider.mass.loop.create_task(self._async_update(do_ping))

        self._last_media_update = time()

    async def _async_update(self, do_ping: bool = False) -> None:
        """Retrieve the latest data from the DMR Device and update the MASS Player.

        :param do_ping: Poll device to check if it is available (online).

        This method is internal as we provide a facade to debounce and throttle those calls.
        """
        assert self.dmr_device is not None

        self.logger.debug(f"Update player {self.name}")

        try:
            # Poll the player for unevented vars
            await self.dmr_device.async_update(do_ping=do_ping)
        except UpnpError as err:
            self.logger.debug("Device unavailable: %r", err)
            await self.async_disconnect()
            return

        # Wait for DLNA events to arrive
        await sleep(1)

        self._update_media()

        # Dynamically determine if we need to poll the Device additionally
        if self._player.state == PlayerState.PLAYING and self._player.needs_poll is False:
            # Not every action produces events per DLNA spec.
            #
            # Therefore we poll every few seconds additionally.
            # This also acts as kind of a stopgap against broken client implementations.
            self._player.needs_poll = True
            self._player.poll_interval = 3

            self.logger.debug(f"Start timed polling for Player {self.name}.")
        elif self._player.state != PlayerState.PLAYING and self._player.needs_poll:
            self._player.needs_poll = False
            self._player.poll_interval = 30

            self.logger.debug(f"Stop timed polling for Player {self.name}.")

        # inform MA of changes and write the player state
        self.provider.mass.players.update(self.id)

    def _update_media(self) -> None:
        """Update attributes of the MA Player from DLNA state."""
        assert self.dmr_device is not None

        prev_url = self._player.current_item_id
        prev_state = self._player.state

        # Update player state
        self._player.state = _TRANSPORT_STATE_TO_PLAYER_STATE[
            self.dmr_device.transport_state or TransportState.STOPPED
        ]

        # Update volume
        self._player.volume_level = int((self.dmr_device.volume_level or 0) * 100)
        self._player.volume_muted = self.dmr_device.is_volume_muted or False

        # Update media
        self._player.current_item_id = self.dmr_device.current_track_uri or ""

        # Update active source
        if self.queue_playing:
            # We are playing the MA queue
            self._player.active_source = self.id
        else:
            # We are playing from an external source
            if self._active_source:
                self._player.active_source = self._active_source

                # Find source in the players source_list to update it dynamically.
                # If we found an item, update its capabilities
                if active_source_item := next(
                    (
                        source
                        for source in self._player.source_list
                        if source.id == self._active_source
                    ),
                    None,
                ):
                    active_source_item.can_play_pause = (
                        self.dmr_device.can_pause and self.dmr_device.can_play
                    )
                    active_source_item.can_next_previous = (
                        self.dmr_device.can_pause and self.dmr_device.can_play
                    )
                    active_source_item.can_seek = self.dmr_device.can_seek_abs_time
                else:
                    # If we did not find one, create it
                    self._player.source_list.append(
                        PlayerSource(
                            id=self._active_source,
                            name=self._active_source.title(),
                            passive=True,
                            can_play_pause=self.dmr_device.can_pause and self.dmr_device.can_play,
                            can_next_previous=self.dmr_device.can_pause
                            and self.dmr_device.can_play,
                            can_seek=self.dmr_device.can_seek_abs_time,
                        )
                    )
            else:
                self._player.active_source = None

            # As we are playing from an external source, sync currently played media to MASS.
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

            # Set elapsed time
            self._player.elapsed_time = (
                float(self.dmr_device.media_position)
                if self.dmr_device.media_position is not None
                else None
            )

            self._player.elapsed_time_last_updated = (
                (self.dmr_device.media_position_updated_at.timestamp())
                if self.dmr_device.media_position_updated_at is not None
                else None
            )

        if prev_state != self._player.state:
            self.logger.debug(f"State of Player {self.name} changed: {self._player.state}")

        if prev_url != self._player.current_item_id:
            self.logger.debug(f"Media of Player {self.name} changed: {self._player.current_media}")

    def _handle_upnp_event(
        self,
        _service: UpnpService,
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

        # Call for an update if an event arrived
        self.update_media()

    async def _async_create_device(self, location: str) -> DmrDevice:
        """Create the device.

        Create the DMR device. Before that, we ensure the device provides the needed
        services for the DMR profile. If those are not found, an error is raised and the player
        is deleted by the provider.
        """
        # If we are connected, disconnect the device first.
        if self.connected:
            await self.async_disconnect()

        # Connect to the base UPNP device.
        upnp_device = await self._upnp_factory.async_create_device(location)

        # Check if the DLNA Device confirms to the DMR profile.
        # We are doing this here and this late to avoid fetching and parsing
        # the description.xml twice.
        if not DmrDevice.is_profile_device(upnp_device):
            self.provider.mass.players.remove(self.id)

            # Will be caught by the provider which will delete the device.
            raise RuntimeError(f"Found incompatible device, removing: {upnp_device.device_type}")

        # Create profile wrapper
        return DmrDevice(upnp_device, self._event_handler)

    async def _async_reinit_device(
        self, location: str, ssdp_headers: CaseInsensitiveDict | None = None
    ) -> None:
        """Reinitialize the device.

        Reinitialize on reboots and config changes. This is signaled via SSDP and needed because,
        technically, devices can change provided services and subdevices while online.
        """
        assert self.dmr_device is not None

        self.logger.debug("Reinitializing device, location: %s", location)

        new_device = await self._upnp_factory.async_create_device(location)
        self.dmr_device.profile_device.reinit(new_device)

        if ssdp_headers:
            self.dmr_device.profile_device.ssdp_headers = ssdp_headers

    @property
    def _possible_sources(self) -> set[str] | None:
        state_var = self.get_device()._state_variable("AVT", "PossiblePlaybackStorageMedia")  # pylint: disable=protected-access
        if not state_var:
            return None

        return _lower_split_commas(state_var.value or "")

    @property
    def _active_source(self) -> str | None:
        state_var = self.get_device()._state_variable("AVT", "PlaybackStorageMedium")  # pylint: disable=protected-access
        if not state_var or not isinstance(state_var.value, str):
            return None

        return state_var.value.strip().lower()
