"""DLNA/uPNP Player provider for Music Assistant.

Most of this code is based on the implementation within Home Assistant:
https://github.com/home-assistant/core/blob/dev/homeassistant/components/dlna_dmr

All rights/credits reserved.
"""

from __future__ import annotations

import logging
from asyncio import Lock
from datetime import timedelta
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, Final, ParamSpec, TypeVar

from async_upnp_client.aiohttp import AiohttpSessionRequester
from async_upnp_client.client_factory import UpnpFactory
from async_upnp_client.exceptions import UpnpError
from music_assistant_models.errors import PlayerUnavailableError

from music_assistant.constants import (
    CONF_ENTRY_CROSSFADE_DURATION,
    CONF_ENTRY_CROSSFADE_FLOW_MODE_REQUIRED,
    CONF_ENTRY_ENABLE_ICY_METADATA,
    CONF_ENTRY_HTTP_PROFILE,
    CONF_ENTRY_OUTPUT_CODEC,
    CONF_ENTRY_OUTPUT_CODEC_DEFAULT_MP3,
    CONF_PLAYERS,
    VERBOSE_LOG_LEVEL,
    create_sample_rates_config_entry,
)
from music_assistant.helpers.didl_lite import create_didl_metadata
from music_assistant.helpers.util import TaskManager
from music_assistant.models.player_provider import PlayerProvider

from .notify_handler import DLNANotifyHandler
from .player import DLNAPlayer
from .ssdp_listener import SSDPListener, get_preferred_location

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from async_upnp_client.const import DeviceOrServiceType, SsdpSource
    from async_upnp_client.profiles.dlna import DmrDevice
    from async_upnp_client.ssdp_listener import SsdpDevice
    from music_assistant_models.config_entries import ConfigEntry, PlayerConfig
    from music_assistant_models.player import PlayerMedia

SSDP_ST_DMR: Final = "urn:schemas-upnp-org:device:MediaRenderer:1"


_DLNAPlayerProviderT = TypeVar("_DLNAPlayerProviderT", bound="DLNAPlayerProvider")
_R = TypeVar("_R")
_P = ParamSpec("_P")


def catch_request_errors(
    func: Callable[Concatenate[_DLNAPlayerProviderT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_DLNAPlayerProviderT, _P], Coroutine[Any, Any, _R | None]]:
    """Catch UpnpError errors."""

    @wraps(func)
    async def wrapper(self: _DLNAPlayerProviderT, *args: _P.args, **kwargs: _P.kwargs) -> _R | None:
        """Catch UpnpError errors and check availability before and after request."""
        player_id = str(kwargs["player_id"] if "player_id" in kwargs else args[0])

        if player_id not in self.dlna_players:
            self.logger.warning(
                "Device %s unknown when trying to call %s", player_id, func.__name__
            )
            return None

        dlna_player = self.dlna_players[player_id]
        if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
            self.logger.debug(
                "Handling command %s for player %s",
                func.__name__,
                dlna_player.name,
            )
        if not dlna_player.available:
            self.logger.warning("Device disappeared when trying to call %s", func.__name__)
            return None
        try:
            return await func(self, *args, **kwargs)
        except UpnpError as err:
            # dlna_player.check_available = True
            if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
                self.logger.exception("Error during call %s: %r", func.__name__, err)
            else:
                self.logger.error("Error during call %s: %r", func.__name__, str(err))
        return None

    return wrapper


class DLNAPlayerProvider(PlayerProvider):  # pylint:disable=abstract-method
    """DLNA Player provider."""

    dlna_players: dict[str, DLNAPlayer]

    _discovery_running: bool = False
    _lock: Lock
    _upnp_factory: UpnpFactory
    _notify_handler: DLNANotifyHandler
    _ssdp_listener: SSDPListener

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self.dlna_players = {}
        self._lock = Lock()

        # silence the async_upnp_client logger
        if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
            logging.getLogger("async_upnp_client").setLevel(logging.DEBUG)
        else:
            logging.getLogger("async_upnp_client").setLevel(self.logger.level + 10)

        requester = AiohttpSessionRequester(self.mass.http_session, with_sleep=True)

        self._upnp_factory = UpnpFactory(requester, non_strict=True)
        self._notify_handler = DLNANotifyHandler(self.mass, requester)
        self._ssdp_listener = SSDPListener(
            search_target=SSDP_ST_DMR,
            session=self.mass.http_session,
            async_callback=self._handle_ssdp_event,
        )

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        self._notify_handler.register()
        await self._ssdp_listener.async_start()

        await super().loaded_in_mass()

    async def unload(self, is_removed: bool = False) -> None:
        """
        Handle unload/close of the provider.

        Called when provider is deregistered (e.g. MA exiting or config reloading).
        """
        self._notify_handler.unregister()
        await self._ssdp_listener.async_stop()

        async with TaskManager(self.mass) as tg:
            for dlna_player in self.dlna_players.values():
                if dlna_player.connected:
                    tg.create_task(dlna_player.async_disconnect())

    async def get_player_config_entries(
        self,
        player_id: str,
    ) -> tuple[ConfigEntry, ...]:
        """Return all (provider/player specific) Config Entries for the given player (if any)."""
        config_entries = (
            *await super().get_player_config_entries(player_id),
            CONF_ENTRY_CROSSFADE_FLOW_MODE_REQUIRED,
            CONF_ENTRY_CROSSFADE_DURATION,
            CONF_ENTRY_OUTPUT_CODEC,
            CONF_ENTRY_HTTP_PROFILE,
            CONF_ENTRY_ENABLE_ICY_METADATA,
            create_sample_rates_config_entry(max_sample_rate=192000, max_bit_depth=24),
        )

        if player_id in self.dlna_players and self.dlna_players[player_id].supports_flac is False:
            config_entries += (CONF_ENTRY_OUTPUT_CODEC_DEFAULT_MP3,)

        return config_entries

    async def on_player_config_change(
        self,
        config: PlayerConfig,
        changed_keys: set[str],
    ) -> None:
        """Call (by config manager) when the configuration of a player changes."""
        if dlna_player := self.dlna_players.get(config.player_id):
            # reset player features based on config values
            dlna_player.update()
        else:
            # run discovery to catch any re-enabled players
            self.mass.create_task(self.discover_players())

    async def poll_player(self, player_id: str) -> None:
        """Poll player for state updates."""
        await self._get_dmr_device(player_id).async_update()

    @catch_request_errors
    async def cmd_stop(self, player_id: str) -> None:
        """Send STOP command to given player."""
        await self._get_dmr_device(player_id).async_stop()

    @catch_request_errors
    async def cmd_play(self, player_id: str) -> None:
        """Send PLAY command to given player."""
        await self._get_dmr_device(player_id).async_play()

    @catch_request_errors
    async def play_media(self, player_id: str, media: PlayerMedia) -> None:
        """Handle PLAY MEDIA on given player."""
        dmr_device = self._get_dmr_device(player_id)

        # always clear queue (by sending stop) first
        if dmr_device.can_stop:
            await self.cmd_stop(player_id)

        didl_metadata = create_didl_metadata(media)
        title = media.title or media.uri
        await dmr_device.async_set_transport_uri(media.uri, title, didl_metadata)

        # Play it
        await dmr_device.async_wait_for_can_play(10)

        await dmr_device.async_play()

    @catch_request_errors
    async def enqueue_next_media(self, player_id: str, media: PlayerMedia) -> None:
        """Handle enqueuing of the next queue item on the player."""
        dmr_device = self._get_dmr_device(player_id)

        didl_metadata = create_didl_metadata(media)
        title = media.title or media.uri
        try:
            await dmr_device.async_set_next_transport_uri(media.uri, title, didl_metadata)
        except UpnpError:
            self.logger.error(
                "Enqueuing the next track failed for player %s - "
                "the player probably doesn't support this. "
                "Enable 'flow mode' for this player.",
                dmr_device.name,
            )
        else:
            self.logger.debug(
                "Enqued next track (%s) to player %s",
                title,
                dmr_device.name,
            )

    @catch_request_errors
    async def cmd_pause(self, player_id: str) -> None:
        """Send PAUSE command to given player."""
        dmr_device = self._get_dmr_device(player_id)

        if dmr_device.can_pause:
            await dmr_device.async_pause()
        else:
            await dmr_device.async_stop()

    @catch_request_errors
    async def cmd_volume_set(self, player_id: str, volume_level: int) -> None:
        """Send VOLUME_SET command to given player."""
        await self._get_dmr_device(player_id).async_set_volume_level(volume_level / 100)

    @catch_request_errors
    async def cmd_volume_mute(self, player_id: str, muted: bool) -> None:
        """Send VOLUME MUTE command to given player."""
        await self._get_dmr_device(player_id).async_mute_volume(muted)

    @catch_request_errors
    async def cmd_seek(self, player_id: str, position: int) -> None:
        """Handle SEEK command for given player."""
        seek_time = timedelta(seconds=position)
        await self._get_dmr_device(player_id).async_seek_abs_time(seek_time)

    async def discover_players(self) -> None:
        """Discover DLNA players on the network."""
        if self._discovery_running:
            return

        try:
            self._discovery_running = True

            await self._ssdp_listener.async_search()
        finally:
            self._discovery_running = False

        def reschedule() -> None:
            self.mass.create_task(self.discover_players())

        # reschedule self once finished
        self.mass.loop.call_later(600, reschedule)

    async def _handle_ssdp_event(
        self,
        ssdp_device: SsdpDevice,
        device_or_service_type: DeviceOrServiceType,
        _ssdp_source: SsdpSource,
    ) -> None:
        """Handle SSDP events."""
        udn = ssdp_device.udn

        # Ignore incompatible players
        if device_or_service_type != SSDP_ST_DMR:
            return

        # ignore Sonos players
        if "rincon" in udn.lower():
            self.logger.debug(f"Ignoring sonos device: {udn}")
            return

        # ignore disabled players
        if self._is_player_disabled(udn):
            self.logger.debug(f"Ignoring disabled player: {udn}")
            return

        if udn not in self.dlna_players:
            # new player detected, setup DLNAPlayer
            self.dlna_players[udn] = DLNAPlayer(
                provider=self,
                udn=udn,
                upnp_factory=self._upnp_factory,
                event_handler=self._notify_handler.event_handler,
            )

        # prefer ipv4 if multiple locations available
        location = get_preferred_location(ssdp_device.locations)

        # get combined search and advertisement headers
        headers = ssdp_device.combined_headers(device_or_service_type)

        # connect the device
        # we handle reconnects and changed devices internally
        try:
            await self.dlna_players[udn].async_connect(location, headers)
        except RuntimeError as err:
            self.logger.error(err)

            self.dlna_players.pop(udn, None)

    def _is_player_disabled(self, udn: str) -> bool:
        conf_key = f"{CONF_PLAYERS}/{udn}/enabled"

        return not self.mass.config.get(conf_key, True)

    def _get_dmr_device(self, player_id: str) -> DmrDevice:
        if player_id not in self.dlna_players:
            raise PlayerUnavailableError

        return self.dlna_players[player_id].get_device()
