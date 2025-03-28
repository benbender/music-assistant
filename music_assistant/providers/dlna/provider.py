"""DLNA provider."""

from __future__ import annotations

import asyncio
import functools
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

from async_upnp_client.aiohttp import AiohttpSessionRequester
from async_upnp_client.client_factory import UpnpFactory
from async_upnp_client.exceptions import UpnpError

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
from music_assistant.models.player_provider import PlayerProvider

from .notify_handler import DLNANotifyHandler
from .player import DLNAPlayer
from .ssdp import SSDPScanner, SsdpServiceInfo, get_preferred_location

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from async_upnp_client.client import UpnpRequester
    from music_assistant_models.config_entries import ConfigEntry, PlayerConfig
    from music_assistant_models.player import PlayerMedia


PLAYER_CONFIG_ENTRIES = (
    CONF_ENTRY_CROSSFADE_FLOW_MODE_REQUIRED,
    CONF_ENTRY_CROSSFADE_DURATION,
    CONF_ENTRY_OUTPUT_CODEC,
    CONF_ENTRY_HTTP_PROFILE,
    CONF_ENTRY_ENABLE_ICY_METADATA,
    create_sample_rates_config_entry(max_sample_rate=192000, max_bit_depth=24),
)


_DLNAPlayerProviderT = TypeVar("_DLNAPlayerProviderT", bound="DLNAPlayerProvider")
_R = TypeVar("_R")
_P = ParamSpec("_P")


def catch_request_errors(
    func: Callable[Concatenate[_DLNAPlayerProviderT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_DLNAPlayerProviderT, _P], Coroutine[Any, Any, _R | None]]:
    """Catch UpnpError errors."""

    @functools.wraps(func)
    async def wrapper(self: _DLNAPlayerProviderT, *args: _P.args, **kwargs: _P.kwargs) -> _R | None:
        """Catch UpnpError errors and check availability before and after request."""
        player_id: str = str(kwargs["player_id"] if "player_id" in kwargs else args[0])
        dlna_player = self.dlnaplayers[player_id]
        if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
            self.logger.debug("Handling command %s for player %s", func.__name__, dlna_player.name)
        if not dlna_player.mass_player.available:
            self.logger.warning("Device disappeared when trying to call %s", func.__name__)
            return None
        try:
            return await func(self, *args, **kwargs)
        except UpnpError as err:
            dlna_player.check_available = True
            if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
                self.logger.exception("Error during call %s: %r", func.__name__, err)
            else:
                self.logger.error("Error during call %s: %r", func.__name__, str(err))
        return None

    return wrapper


class DLNAPlayerProvider(PlayerProvider):
    """DLNA Player provider."""

    dlnaplayers: dict[str, DLNAPlayer] = {}
    _discovery_running: bool = False

    lock: asyncio.Lock
    requester: UpnpRequester
    upnp_factory: UpnpFactory
    notify_handler: DLNANotifyHandler

    ssdp_scanner: SSDPScanner

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self.dlnaplayers: dict[str, DLNAPlayer]
        self.lock = asyncio.Lock()

        # silence the async_upnp_client logger
        if self.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
            logging.getLogger("async_upnp_client").setLevel(logging.DEBUG)
        else:
            logging.getLogger("async_upnp_client").setLevel(self.logger.level + 10)

        self.requester = AiohttpSessionRequester(self.mass.http_session, with_sleep=True)
        self.upnp_factory = UpnpFactory(self.requester, non_strict=True)
        self.notify_handler = DLNANotifyHandler(self.mass, self.requester)
        self.ssdp_scanner = SSDPScanner(
            session=self.mass.http_session, on_discover=self._player_discovered
        )

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        self.notify_handler.start()
        await self.ssdp_scanner.async_start()

        await super().loaded_in_mass()

    async def unload(self, is_removed: bool = False) -> None:
        """
        Handle unload/close of the provider.

        Called when provider is deregistered (e.g. MA exiting or config reloading).
        """
        # if mass is currently streaming to a device,
        # we want to stop that stream to avoid hickups on shutdown
        await asyncio.gather(
            *(player.async_disconnect(True) for player in self.dlnaplayers.values())
        )

        await self.ssdp_scanner.async_stop()
        self.notify_handler.stop()

    async def get_player_config_entries(
        self,
        player_id: str,
    ) -> tuple[ConfigEntry, ...]:
        """Return all (provider/player specific) Config Entries for the given player (if any)."""
        config_entries: tuple[ConfigEntry, ...] = PLAYER_CONFIG_ENTRIES

        if player_id in self.dlnaplayers and not self.dlnaplayers[player_id].supports_flac:
            config_entries += (CONF_ENTRY_OUTPUT_CODEC_DEFAULT_MP3,)

        config_entries += await super().get_player_config_entries(player_id)

        return config_entries

    async def on_player_config_change(
        self,
        config: PlayerConfig,
        changed_keys: set[str],
    ) -> None:
        """Call (by config manager) when the configuration of a player changes."""
        if dlna_player := self.dlnaplayers.get(config.player_id):
            # reset player features based on config values
            await dlna_player.async_update()
        else:
            # run discovery to catch any re-enabled players
            self.mass.create_task(self.discover_players())

    async def poll_player(self, player_id: str) -> None:
        """Poll player for state updates."""
        dlnaplayer = self.dlnaplayers[player_id]

        self.logger.debug("Polling player %s", dlnaplayer.player_id)

    @catch_request_errors
    async def cmd_stop(self, player_id: str) -> None:
        """Send STOP command to given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        await dlna_player.dmr_device.async_stop()

    @catch_request_errors
    async def cmd_play(self, player_id: str) -> None:
        """Send PLAY command to given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        await dlna_player.dmr_device.async_play()

    @catch_request_errors
    async def cmd_next(self, player_id: str) -> None:
        """Send NEXT command to given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        await dlna_player.dmr_device.async_next()

    @catch_request_errors
    async def cmd_previous(self, player_id: str) -> None:
        """Send PREVIOUS command to given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        await dlna_player.dmr_device.async_previous()

    @catch_request_errors
    async def play_media(self, player_id: str, media: PlayerMedia) -> None:
        """Handle PLAY MEDIA on given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        # always clear queue (by sending stop) first
        if dlna_player.dmr_device.can_stop:
            await self.cmd_stop(player_id)

        didl_metadata = create_didl_metadata(media)
        title = media.title or media.uri

        await dlna_player.dmr_device.async_set_transport_uri(media.uri, title, didl_metadata)

        # Play it
        await dlna_player.dmr_device.async_wait_for_can_play(10)

        await dlna_player.dmr_device.async_play()

    @catch_request_errors
    async def enqueue_next_media(self, player_id: str, media: PlayerMedia) -> None:
        """Handle enqueuing of the next queue item on the player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        didl_metadata = create_didl_metadata(media)
        title = media.title or media.uri
        try:
            await dlna_player.dmr_device.async_set_next_transport_uri(
                media.uri, title, didl_metadata
            )
        except UpnpError:
            self.logger.error(
                "Enqueuing the next track failed for player %s - "
                "the player probably doesn't support this. "
                "Enable 'flow mode' for this player.",
                dlna_player.name,
            )
        else:
            self.logger.debug(
                "Enqued next track (%s) to player %s",
                title,
                dlna_player.name,
            )

    @catch_request_errors
    async def cmd_pause(self, player_id: str) -> None:
        """Send PAUSE command to given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        if dlna_player.dmr_device.can_pause:
            await dlna_player.dmr_device.async_pause()
        else:
            await dlna_player.dmr_device.async_stop()

    @catch_request_errors
    async def cmd_volume_set(self, player_id: str, volume_level: int) -> None:
        """Send VOLUME_SET command to given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        await dlna_player.dmr_device.async_set_volume_level(volume_level / 100)

    @catch_request_errors
    async def cmd_volume_mute(self, player_id: str, muted: bool) -> None:
        """Send VOLUME MUTE command to given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        await dlna_player.dmr_device.async_mute_volume(muted)

    @catch_request_errors
    async def cmd_seek(self, player_id: str, position: int) -> None:
        """Handle SEEK command for given player."""
        dlna_player = self.dlnaplayers[player_id]
        assert dlna_player.dmr_device is not None

        seek_time = timedelta(seconds=position)
        await dlna_player.dmr_device.async_seek_abs_time(seek_time)

    async def discover_players(self) -> None:
        """Discover DLNA players on the network."""
        if self._discovery_running:
            return

        try:
            self._discovery_running = True

            await self.ssdp_scanner.async_scan()
        finally:
            self._discovery_running = False

        def reschedule() -> None:
            self.mass.create_task(self.discover_players())

        # reschedule self once finished
        self.mass.loop.call_later(600, reschedule)

    async def _player_discovered(self, discovery_info: SsdpServiceInfo) -> None:
        """Handle discovered DLNA player."""
        async with self.lock:
            # if multiple locations are given for the player, prefer ipv4 because
            # of broken ipv6-stacks out there. If only v6 is available, use it though.
            description_url = get_preferred_location(discovery_info.ssdp_all_locations)

            # new player detected, setup our DLNAPlayer wrapper
            conf_key = f"{CONF_PLAYERS}/{discovery_info.ssdp_udn}/enabled"
            enabled = self.mass.config.get(conf_key, True)

            # ignore disabled players
            if not enabled:
                self.logger.debug("Ignoring disabled player: %s", discovery_info.ssdp_udn)
                return

            self.dlnaplayers[discovery_info.ssdp_udn] = DLNAPlayer(
                self,
                discovery_info,
                self.upnp_factory,
                self.notify_handler.event_handler,
            )

            await self.dlnaplayers[discovery_info.ssdp_udn].async_connect(description_url)
