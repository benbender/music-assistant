"""DLNA/uPNP Player provider for Music Assistant.

Most of this code is based on the implementation within Home Assistant:
https://github.com/home-assistant/core/blob/dev/homeassistant/components/dlna_dmr

All rights/credits reserved.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from async_upnp_client.profiles.dlna import DmrDevice, TransportState
from music_assistant_models.enums import PlayerState

if TYPE_CHECKING:
    from music_assistant_models.player import Player


@dataclass
class DLNAPlayer:
    """Class that holds all dlna variables for a player."""

    udn: str  # = player_id
    player: Player  # mass player
    description_url: str  # last known location (description.xml) url

    device: DmrDevice | None = None
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )  # Held when connecting or disconnecting the device
    force_poll: bool = False
    ssdp_connect_failed: bool = False

    # Track BOOTID in SSDP advertisements for device changes
    bootid: int | None = None
    last_seen: float = field(default_factory=time.time)
    last_command: float = field(default_factory=time.time)

    def update_attributes(self) -> None:
        """Update attributes of the MA Player from DLNA state."""
        # generic attributes

        if self.available:
            self.player.available = True
            self.player.name = self.device.name
            self.player.volume_level = int((self.device.volume_level or 0) * 100)
            self.player.volume_muted = self.device.is_volume_muted or False
            self.player.state = self.get_state(self.device)
            self.player.current_item_id = self.device.current_track_uri or ""
            if self.player.player_id in self.player.current_item_id:
                self.player.active_source = self.player.player_id
            elif "spotify" in self.player.current_item_id:
                self.player.active_source = "spotify"
            elif self.player.current_item_id.startswith("http"):
                self.player.active_source = "http"
            else:
                # TODO: handle other possible sources here
                self.player.active_source = None
            if self.device.media_position:
                # only update elapsed_time if the device actually reports it
                self.player.elapsed_time = float(self.device.media_position)
                if self.device.media_position_updated_at is not None:
                    self.player.elapsed_time_last_updated = (
                        self.device.media_position_updated_at.timestamp()
                    )
        else:
            # device is unavailable
            self.player.available = False

    @property
    def available(self) -> bool:
        """Device is available when we have a connection to it."""
        return self.device is not None and self.device.profile_device.available

    @staticmethod
    def get_state(device: DmrDevice) -> PlayerState:
        """Return current PlayerState of the player."""
        if device.transport_state is None:
            return PlayerState.IDLE
        if device.transport_state in (
            TransportState.PLAYING,
            TransportState.TRANSITIONING,
        ):
            return PlayerState.PLAYING
        if device.transport_state in (
            TransportState.PAUSED_PLAYBACK,
            TransportState.PAUSED_RECORDING,
        ):
            return PlayerState.PAUSED
        if device.transport_state == TransportState.VENDOR_DEFINED:
            # Unable to map this state to anything reasonable, fallback to idle
            return PlayerState.IDLE

        return PlayerState.IDLE
