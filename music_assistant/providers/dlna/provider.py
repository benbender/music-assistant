"""DLNA/uPNP Player provider for Music Assistant.

Most of this code is based on the implementation within Home Assistant:
https://github.com/home-assistant/core/blob/dev/homeassistant/components/dlna_dmr

All rights/credits reserved.
"""

from __future__ import annotations

import functools
import logging
import time
from asyncio import Lock
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

from async_upnp_client.aiohttp import AiohttpSessionRequester
from async_upnp_client.client_factory import UpnpFactory
from async_upnp_client.exceptions import UpnpError
from async_upnp_client.search import async_search
from music_assistant_models.errors import PlayerUnavailableError

from music_assistant.constants import (
    CONF_ENTRY_CROSSFADE_DURATION,
    CONF_ENTRY_CROSSFADE_FLOW_MODE_REQUIRED,
    CONF_ENTRY_ENABLE_ICY_METADATA,
    CONF_ENTRY_FLOW_MODE_DEFAULT_ENABLED,
    CONF_ENTRY_HTTP_PROFILE,
    CONF_ENTRY_OUTPUT_CODEC,
    CONF_PLAYERS,
    VERBOSE_LOG_LEVEL,
    create_sample_rates_config_entry,
)
from music_assistant.helpers.didl_lite import create_didl_metadata
from music_assistant.helpers.util import TaskManager
from music_assistant.models.player_provider import PlayerProvider

from .notify_handler import DLNANotifyHandler
from .player import DLNAPlayer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from async_upnp_client.utils import CaseInsensitiveDict
    from music_assistant_models.config_entries import ConfigEntry, PlayerConfig
    from music_assistant_models.player import PlayerMedia


PLAYER_CONFIG_ENTRIES = (
    CONF_ENTRY_CROSSFADE_FLOW_MODE_REQUIRED,
    CONF_ENTRY_CROSSFADE_DURATION,
    CONF_ENTRY_OUTPUT_CODEC,
    CONF_ENTRY_HTTP_PROFILE,
    CONF_ENTRY_ENABLE_ICY_METADATA,
    # enable flow mode by default because
    # most dlna players do not support enqueueing
    CONF_ENTRY_FLOW_MODE_DEFAULT_ENABLED,
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
            dlna_player.force_poll = True
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

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        self._notify_handler.register()

        await super().loaded_in_mass()

    async def unload(self, is_removed: bool = False) -> None:
        """
        Handle unload/close of the provider.

        Called when provider is deregistered (e.g. MA exiting or config reloading).
        """
        self._notify_handler.unregister()

        async with TaskManager(self.mass) as tg:
            for dlna_player in self.dlna_players.values():
                tg.create_task(dlna_player.async_disconnect())

    async def get_player_config_entries(
        self,
        player_id: str,
    ) -> tuple[ConfigEntry, ...]:
        """Return all (provider/player specific) Config Entries for the given player (if any)."""
        base_entries = await super().get_player_config_entries(player_id)

        return base_entries + PLAYER_CONFIG_ENTRIES

    async def on_player_config_change(
        self,
        config: PlayerConfig,
        changed_keys: set[str],
    ) -> None:
        """Call (by config manager) when the configuration of a player changes."""
        if dlna_player := self.dlna_players.get(config.player_id):
            # reset player features based on config values
            await dlna_player.async_update()
        else:
            # run discovery to catch any re-enabled players
            self.mass.create_task(self.discover_players())

    @catch_request_errors
    async def cmd_stop(self, player_id: str) -> None:
        """Send STOP command to given player."""
        dlna_player = self.dlna_players[player_id]
        assert dlna_player.dmr_device is not None
        await dlna_player.dmr_device.async_stop()

    @catch_request_errors
    async def cmd_play(self, player_id: str) -> None:
        """Send PLAY command to given player."""
        dlna_player = self.dlna_players[player_id]
        assert dlna_player.dmr_device is not None
        await dlna_player.dmr_device.async_play()

    @catch_request_errors
    async def play_media(self, player_id: str, media: PlayerMedia) -> None:
        """Handle PLAY MEDIA on given player."""
        dlna_player = self.dlna_players[player_id]
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
        dlna_player = self.dlna_players[player_id]

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
        dlna_player = self.dlna_players[player_id]
        assert dlna_player.dmr_device is not None
        if dlna_player.dmr_device.can_pause:
            await dlna_player.dmr_device.async_pause()
        else:
            await dlna_player.dmr_device.async_stop()

    @catch_request_errors
    async def cmd_volume_set(self, player_id: str, volume_level: int) -> None:
        """Send VOLUME_SET command to given player."""
        dlna_player = self.dlna_players[player_id]
        assert dlna_player.dmr_device is not None
        await dlna_player.dmr_device.async_set_volume_level(volume_level / 100)

    @catch_request_errors
    async def cmd_volume_mute(self, player_id: str, muted: bool) -> None:
        """Send VOLUME MUTE command to given player."""
        dlna_player = self.dlna_players[player_id]
        assert dlna_player.dmr_device is not None
        await dlna_player.dmr_device.async_mute_volume(muted)

    async def poll_player(self, player_id: str) -> None:
        """Poll player for state updates."""
        dlna_player = self.dlna_players[player_id]

        # # try to reconnect the device if the connection was lost
        # if not dlna_player.dmr_device:
        #     if not dlna_player.force_poll:
        #         return
        #     try:
        #         await dlna_player.async_connect()
        #     except UpnpError as err:
        #         raise PlayerUnavailableError from err

        assert dlna_player.dmr_device is not None

        try:
            now = time.time()
            do_ping = dlna_player.force_poll or (now - dlna_player.last_seen) > 60
            with suppress(ValueError):
                await dlna_player.async_update(do_ping=do_ping)
            dlna_player.last_seen = now if do_ping else dlna_player.last_seen
        except UpnpError as err:
            self.logger.debug("Device unavailable: %r", err)
            await dlna_player.async_disconnect()
            raise PlayerUnavailableError from err
        finally:
            dlna_player.force_poll = False

    async def discover_players(self, use_multicast: bool = False) -> None:
        """Discover DLNA players on the network."""
        if self._discovery_running:
            return
        try:
            self._discovery_running = True
            self.logger.debug("DLNA discovery started...")
            discovered_devices: set[str] = set()

            async def on_response(discovery_info: CaseInsensitiveDict) -> None:
                """Process discovered device from ssdp search."""
                ssdp_st: str = discovery_info.get("st", discovery_info.get("nt"))
                if not ssdp_st:
                    return

                if "MediaRenderer" not in ssdp_st:
                    # we're only interested in MediaRenderer devices
                    return

                ssdp_usn: str = discovery_info["usn"]
                ssdp_udn: str | None = discovery_info.get("_udn")
                if not ssdp_udn and ssdp_usn.startswith("uuid:"):
                    ssdp_udn = ssdp_usn.split("::")[0]

                if ssdp_udn in discovered_devices:
                    # already processed this device
                    return

                if ssdp_udn:
                    # ignore Sonos devices
                    if "rincon" in ssdp_udn.lower():
                        return

                    discovered_devices.add(ssdp_udn)

                    await self._device_discovered(ssdp_udn, discovery_info["location"])

            await async_search(on_response)

        finally:
            self._discovery_running = False

        def reschedule() -> None:
            self.mass.create_task(self.discover_players(use_multicast=not use_multicast))

        # reschedule self once finished
        self.mass.loop.call_later(300, reschedule)

    async def _device_discovered(self, udn: str, location: str) -> None:
        """Handle discovered DLNA player."""
        async with self._lock:
            if dlna_player := self.dlna_players.get(udn):
                # existing player
                await dlna_player.async_connect(location)
            else:
                # ignore disabled players
                if self._is_player_disabled(udn):
                    self.logger.debug("Ignoring disabled player: %s", udn)
                    return

                # new player detected, setup our DLNAPlayer wrapper
                self.dlna_players[udn] = DLNAPlayer(
                    provider=self,
                    udn=udn,
                    upnp_factory=self._upnp_factory,
                    event_handler=self._notify_handler.event_handler,
                )

            await self.dlna_players[udn].async_connect(location)

    def _is_player_disabled(self, udn: str) -> bool:
        conf_key = f"{CONF_PLAYERS}/{udn}/enabled"

        return not self.mass.config.get(conf_key, True)
