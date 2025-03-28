"""DLNA player."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from contextlib import suppress
from time import time
from typing import TYPE_CHECKING

from async_upnp_client.device_updater import DeviceUpdater
from async_upnp_client.exceptions import UpnpError, UpnpResponseError
from async_upnp_client.profiles.dlna import DmrDevice, TransportState, split_commas, str_to_time
from music_assistant_models.enums import (
    MediaType,
    PlayerState,
)
from music_assistant_models.player import (
    DeviceInfo,
    Player,
    PlayerFeature,
    PlayerSource,
    PlayerType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from async_upnp_client.client import UpnpEventHandler, UpnpService, UpnpStateVariable
    from async_upnp_client.client_factory import UpnpFactory

    from music_assistant import MusicAssistant

    from .ssdp import SsdpServiceInfo


# Map UPnP class to media_player media_content_type
MEDIA_TYPE_MAP: Mapping[str, MediaType] = {
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
_TRANSPORT_STATE_TO_MEDIA_PLAYER_STATE: Mapping[TransportState, PlayerState] = {
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

_LOGGER = logging.getLogger(__name__)


class DLNAPlayer:
    """Class that holds all dlna variables for a player."""

    # Last known URL for the device, used when adding this entity to hass to try
    # to connect before SSDP has rediscovered it, or when SSDP discovery fails.
    location: str

    check_available: bool = False
    _ssdp_connect_failed: bool = False

    mass_player: Player
    event_handler: UpnpEventHandler
    dmr_device: DmrDevice | None = None
    updater: DeviceUpdater | None = None

    # Held when connecting or disconnecting the device
    _device_lock: asyncio.Lock
    _sources: Mapping[str, PlayerSource] = {}
    supports_flac: bool = True

    def __init__(
        self,
        mass: MusicAssistant,
        instance_id: str,
        discovery_info: SsdpServiceInfo,
        upnp_factory: UpnpFactory,
        event_handler: UpnpEventHandler,
    ) -> None:
        """Init."""
        self.mass = mass
        self._device_lock = asyncio.Lock()
        self.upnp_factory = upnp_factory
        self.event_handler = event_handler

        self.mass_player = Player(
            player_id=discovery_info.ssdp_udn,
            provider=instance_id,
            type=PlayerType.PLAYER,
            name=discovery_info.ssdp_udn,
            available=False,
            needs_poll=False,
            device_info=DeviceInfo(ip_address=discovery_info.ssdp_location),
        )

    async def async_connect(self, location: str) -> None:
        """Connect the player and subscribe to events."""
        async with self._device_lock:
            _LOGGER.info("Connecting player %s", self.player_id)

            # Connect to the base UPNP device
            upnp_device = await self.upnp_factory.async_create_device(location)

            # Create profile wrapper
            self.dmr_device = DmrDevice(upnp_device, self.event_handler)

            self.updater = DeviceUpdater(self.dmr_device.device, self.upnp_factory)

            self.mass_player.name = self.dmr_device.name
            self.mass_player.device_info = DeviceInfo(
                ip_address=self.dmr_device.device.device_url,
                model=self.dmr_device.model_name.strip(),
                model_id=self.dmr_device.model_number.strip()
                if self.dmr_device.model_number
                else None,
                manufacturer=self.dmr_device.manufacturer.strip(),
            )

            # Subscribe to event notifications
            try:
                self.dmr_device.on_event = self._handle_event
                await self.dmr_device.async_subscribe_services(auto_resubscribe=True)
            except UpnpResponseError as err:
                # Device rejected subscription request. This is OK, variables
                # will be polled instead.
                _LOGGER.info("Device rejected subscription: %r", err)
            except UpnpError as err:
                # Don't leave the device half-constructed
                self.dmr_device.on_event = None
                # dlna_player.device = None
                _LOGGER.info("Error while subscribing during device connect: %r", err)
                raise

            self.mass_player.available = self.dmr_device.device.available

            # needs to be called before registering the device
            self._update_supported_features()

            await self.async_update()

            await self.mass.players.register_or_update(self.mass_player)

    async def async_disconnect(self, stop_streams: bool = False) -> None:
        """
        Destroy connections to the device now that it's not available.

        Also call when removing this entity from MA to clean up connections.
        """
        _LOGGER.info("Disonnecting player %s", self.player_id)

        async with self._device_lock:
            if not self.dmr_device:
                _LOGGER.info("Disconnecting from device that's not connected")
                return

            _LOGGER.info("Disconnecting from %s", self.dmr_device.name)

            if (
                stop_streams
                and self.mass_player.state == PlayerState.PLAYING
                and (
                    self.mass_player.current_item_id
                    and self.player_id in self.mass_player.current_item_id
                )
                and self.dmr_device
                and self.dmr_device.can_stop
            ):
                with suppress(RuntimeError):
                    await self.dmr_device.async_stop()

            self.dmr_device.on_event = None
            await self.dmr_device.async_unsubscribe_services()

            if self.updater:
                await self.updater.async_stop()

    async def async_update(self, do_ping: bool = False) -> None:
        """Retrieve the latest data.

        :param do_ping: Poll device to check if it is available (online).
        """
        _LOGGER.info("async_update %s", self.player_id)

        if not self.dmr_device:
            try:
                await self.async_connect(self.location)
            except UpnpError:
                return

        assert self.dmr_device is not None

        try:
            await self.dmr_device.async_update(do_ping=self.check_available or do_ping)
        except UpnpError as err:
            _LOGGER.debug("Device unavailable: %r", err)
            await self.async_disconnect()
            return
        finally:
            self.check_available = False

        self._update_supported_features()

    @property
    def player_id(self) -> str:
        """Player id."""
        return self.mass_player.player_id

    @property
    def name(self) -> str:
        """Player name."""
        return self.mass_player.name

    @property
    def available(self) -> bool:
        """Device is available when we have a connection to it."""
        return self.dmr_device is not None and self.dmr_device.profile_device.available

    def _handle_event(  # noqa: PLR0915
        self,
        service: UpnpService,
        state_variables: Sequence[UpnpStateVariable],
    ) -> None:
        """Handle state variable(s) changed event from DLNA device."""
        if not state_variables:
            # Indicates a failure to resubscribe, check if device is still available
            self.check_available = True
            return

        assert self.dmr_device is not None

        _refresh_player = False

        if service.service_id == "urn:upnp-org:serviceId:ConnectionManager":
            for state_variable in state_variables:
                if state_variable.name == "SinkProtocolInfo":
                    # Does this device support flac?
                    self.supports_flac = False

                    for item in split_commas(state_variable.value):
                        if "audio/flac" in item.lower():
                            self.supports_flac = True
                            break

                    _LOGGER.info("SinkProtocolInfo supports_flac %s", self.supports_flac)

        elif service.service_id == "urn:upnp-org:serviceId:RenderingControl":
            for state_variable in state_variables:
                if state_variable.name == "Volume":
                    self.mass_player.volume_level = int((self.dmr_device.volume_level or 0) * 100)
                    _refresh_player = True

                elif state_variable.name == "Mute":
                    self.mass_player.volume_muted = self.dmr_device.is_volume_muted or False
                    _refresh_player = True

        elif service.service_id == "urn:upnp-org:serviceId:AVTransport":
            for state_variable in state_variables:
                if state_variable.name in ("CurrentTrackMetaData", "AVTransportURIMetaData"):
                    # sync current_media
                    self.mass_player.set_current_media(
                        uri=self.dmr_device.current_track_uri or "",
                        media_type=MEDIA_TYPE_MAP.get(self.dmr_device.media_class or "object")
                        or MediaType.UNKNOWN,
                        title=self.dmr_device.media_title,
                        artist=self.dmr_device.media_artist,
                        album=self.dmr_device.media_album_name,
                        image_url=self.dmr_device.media_image_url,
                        duration=self.dmr_device.media_duration,
                    )

                    self.mass.loop.create_task(self._update_current_position())

                    _refresh_player = True

                    # elif state_variable.name == "PossiblePlaybackStorageMedia":
                    #     self.mass_player.source_list.clear()

                    #     if state_variable.value:
                    #         for source_name in split_commas(state_variable.value):
                    #             if source_name.lower() in ("unknown", "not_implemented", "none"):
                    #                 continue

                    #             self.mass_player.source_list.append(
                    #                 PlayerSource(
                    #                     id=source_name.lower(),
                    #                     name=source_name.title(),
                    #                     passive=True,
                    #                     can_play_pause=False,
                    #                     can_next_previous=False,
                    #                     can_seek=False,
                    #                 )
                    #             )

                    #     # _LOGGER.warning("self._sources %s", self._sources)

                    _refresh_player = True

                elif state_variable.name == "PlaybackStorageMedium":
                    active_source = None

                    if (
                        self.mass_player.current_item_id
                        and self.player_id in self.mass_player.current_item_id
                    ):
                        active_source = self.player_id

                    elif state_variable.value:
                        self.mass_player.source_list.clear()

                        active_source = state_variable.value.lower()

                        if active_source in ("unknown", "none", "un_known", ""):
                            active_source = None

                        if active_source:
                            self.mass_player.source_list.append(
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

                    self.mass_player.active_source = active_source

                    _refresh_player = True

                elif state_variable.name in ("AVTransportURI", "CurrentTrackURI"):
                    if self.mass_player.current_media:
                        self.mass_player.current_media.uri = state_variable.value or ""
                        _refresh_player = True

                elif state_variable.name in ("CurrentTrackDuration", "CurrentMediaDuration"):
                    if self.mass_player.current_media:
                        self.mass_player.current_media.duration = (
                            self.dmr_device.media_duration or None
                        )
                        self.mass.loop.create_task(self._update_current_position())
                        _refresh_player = True
                    else:
                        _LOGGER.warning(
                            "tried to update media_duration without current_media %s",
                            self.dmr_device.media_duration,
                        )

                # elif state_variable.name == "RelativeTimePosition":
                #     self.mass_player.elapsed_time = self.dmr_device.media_position or None

                #     if self.dmr_device.media_position_updated_at is not None:
                #         self.mass_player.elapsed_time_last_updated = datetime_from_utc_to_local(
                #             self.dmr_device.media_position_updated_at
                #         ).timestamp()
                #     else:
                #         self.mass_player.elapsed_time_last_updated = None

                #     _refresh_player = True

                elif state_variable.name == "TransportState":
                    self.mass_player.state = _TRANSPORT_STATE_TO_MEDIA_PLAYER_STATE[
                        self.dmr_device.transport_state or TransportState.STOPPED
                    ]

                    if state_variable.name == "TransportState" and state_variable.value in (
                        TransportState.PLAYING,
                        TransportState.PAUSED_PLAYBACK,
                    ):
                        self.mass.loop.create_task(self._update_current_position())

                    _refresh_player = True

        if _refresh_player:
            self.mass_player.available = self.dmr_device.profile_device.available
            self.mass.players.update(self.player_id)

    async def _update_current_position(self) -> None:
        """GetPositionInfo is not evented and therefore has to be called manually."""
        assert self.dmr_device is not None

        if self.dmr_device.transport_state in (
            TransportState.PLAYING,
            TransportState.PAUSED_PLAYBACK,
        ):
            service = self.dmr_device.device.service("urn:schemas-upnp-org:service:AVTransport:1")
            if not service:
                return

            action = service.action("GetPositionInfo")

            if not action:
                return

            result = await action.async_call(InstanceID=0)

            # the replace()-call fixes some broken client implementations on LinkPlay-devices…
            # shouldn't harm otherwise.
            elapsed_time = str_to_time(result["RelTime"].replace("-", ""))
            if elapsed_time is None:
                _LOGGER.error("borked RelTime %s", result["RelTime"])
                return

            # only update elapsed_time if the device actually reports it
            self.mass_player.elapsed_time = elapsed_time.seconds
            self.mass_player.elapsed_time_last_updated = time()

            self.mass.players.update(self.player_id)
        elif self.dmr_device.transport_state in (
            TransportState.STOPPED,
            TransportState.TRANSITIONING,
        ):
            self.mass_player.elapsed_time = None
            self.mass_player.elapsed_time_last_updated = None

    def _update_supported_features(self) -> None:
        assert self.dmr_device is not None

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

        self.mass_player.supported_features = supported_features


# def datetime_from_utc_to_local(utc_datetime: datetime) -> datetime:
#     """Convert a utc datetime to the local tz."""
#     now_timestamp = time()
#     offset = datetime.fromtimestamp(now_timestamp) - datetime.utcfromtimestamp(now_timestamp)

#     return utc_datetime + offset
