"""DLNA/uPNP Player provider for Music Assistant.

Most of this code is based on the implementation within Home Assistant:
https://github.com/home-assistant/core/blob/dev/homeassistant/components/dlna_dmr

All rights/credits reserved.
"""

from __future__ import annotations

from asyncio import Lock
from time import time
from typing import TYPE_CHECKING, Any

from async_upnp_client.exceptions import UpnpError, UpnpResponseError
from async_upnp_client.profiles.dlna import DmrDevice, TransportState
from music_assistant_models.enums import PlayerFeature, PlayerState, PlayerType
from music_assistant_models.errors import PlayerUnavailableError
from music_assistant_models.player import DeviceInfo, Player

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


class DLNAPlayer:
    """Class that holds all dlna variables for a player."""

    dmr_device: DmrDevice | None = None

    force_poll: bool = False
    ssdp_connect_failed: bool = False
    check_available: bool = False
    last_seen: float = time()

    _player: Player  # mass player

    # Held when connecting or disconnecting the device
    _lock: Lock

    def __init__(
        self,
        provider: DLNAPlayerProvider,
        udn: str,
        upnp_factory: UpnpFactory,
        event_handler: UpnpEventHandler,
    ) -> None:
        """Initialize the DLNA player."""
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

        # register device
        self.provider.mass.loop.create_task(
            self.provider.mass.players.register_or_update(self._player)
        )

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
        """Device is connected when a DmrDevice exists.

        Check self.available to see if the device is available.
        """
        return self.dmr_device is not None

    @property
    def available(self) -> bool:
        """Device is available when we have a connection to it."""
        return (
            self.connected
            and self.dmr_device is not None
            and self.dmr_device.profile_device.available
        )

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

    async def async_connect(
        self, location: str, headers: CaseInsensitiveDict | None = None
    ) -> None:
        """Connect DLNA/DMR Device."""
        async with self._lock:
            try:
                self.logger.debug("Connecting to device at %s", location)

                do_reinit = False

                # handle first connect
                if self.dmr_device is None:
                    self.logger.debug("First connect, creating dmr device")

                    # Connect to the base UPNP device
                    upnp_device = await self._upnp_factory.async_create_device(location)

                    # Create profile wrapper
                    self.dmr_device = DmrDevice(upnp_device, self._event_handler)
                elif headers:
                    # Handle BOOTID.UPNP.ORG.
                    boot_id = headers.get("BOOTID.UPNP.ORG")
                    device_boot_id = self.dmr_device.profile_device.ssdp_headers.get(
                        "BOOTID.UPNP.ORG"
                    )
                    if boot_id and boot_id != device_boot_id:
                        self.logger.info(
                            "Found changed boot_id: %s, old boot_id: %s",
                            boot_id,
                            device_boot_id,
                        )
                        do_reinit = True

                    # Handle CONFIGID.UPNP.ORG.
                    config_id = headers.get("CONFIGID.UPNP.ORG")
                    device_config_id = self.dmr_device.profile_device.ssdp_headers.get(
                        "CONFIGID.UPNP.ORG"
                    )
                    if config_id and config_id != device_config_id:
                        self.logger.info(
                            "Found changed config_id: %s, old config_id: %s",
                            config_id,
                            device_config_id,
                        )
                        do_reinit = True

                    if "LOCATION" in headers:
                        location = str(headers.get("LOCATION"))

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
                        await self._reinit_device(location, headers)
                    else:
                        # this is only needed when the connect is triggered manually
                        # without providing headers
                        await self._create_device(location)

                # Subscribe to event notifications
                self.dmr_device.on_event = self._handle_event

                # subscribe to DLNA-events. Automatically resubscribes if already subscribed.
                await self.dmr_device.async_subscribe_services(auto_resubscribe=True)

                await self.async_update()
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
            else:
                # connect was successful, update device info
                self._player.device_info = DeviceInfo(
                    model=self.dmr_device.model_name,
                    ip_address=self.dmr_device.profile_device.device_url or location,
                    manufacturer=self.dmr_device.manufacturer,
                )
                self._player.powered = True
                self.dmr_device.profile_device.available = True

    async def async_disconnect(self) -> None:
        """
        Destroy connections to the device now that it's not available.

        Also call when removing this entity from MA to clean up connections.
        """
        async with self._lock:
            if not self.dmr_device:
                self.logger.debug("Disconnecting from device that's not connected")
                return

            self.logger.debug("Disconnecting from %s", self.dmr_device.name)

            self.dmr_device.on_event = None
            old_device = self.dmr_device
            self.dmr_device = None
            await old_device.async_unsubscribe_services()

    async def async_update(self, do_ping: bool = False) -> None:
        """Retrieve the latest data.

        :param do_ping: Poll device to check if it is available (online).
        """
        self.logger.debug("Updating player")

        assert self.dmr_device is not None

        # self.provider.mass.loop.call_soon()

        if self.force_poll:
            try:
                await self.dmr_device.async_update(do_ping=self.check_available or do_ping)
            except UpnpError as err:
                self.logger.debug("Device unavailable: %r", err)
                await self.async_disconnect()
                return
            finally:
                self.check_available = False

        self._update_supported_features()
        self._update_attributes()

        self.provider.mass.players.update(self.id)

    def get_device(self) -> DmrDevice:
        """Get DMR device."""
        if not self.available or self.dmr_device is None:
            raise PlayerUnavailableError

        return self.dmr_device

    def _handle_event(
        self,
        service: UpnpService,
        state_variables: Sequence[UpnpStateVariable[Any]],
    ) -> None:
        """Handle state variable(s) changed event from DLNA device."""
        if not state_variables:
            # Indicates a failure to resubscribe, check if device is still available
            self.force_poll = True
            return

        if service.service_id == "urn:upnp-org:serviceId:AVTransport":
            for state_variable in state_variables:
                # Force a state refresh when player begins or pauses playback
                # to update the position info.
                if state_variable.name == "TransportState" and state_variable.value in (
                    TransportState.PLAYING,
                    TransportState.PAUSED_PLAYBACK,
                ):
                    self.force_poll = True
                    self.provider.mass.create_task(self.async_update())
                    self.logger.debug(
                        "Received new state from event for Player %s: %s",
                        self._player.display_name,
                        state_variable.value,
                    )

        self.provider.mass.create_task(self._update_player())

    async def _reinit_device(self, location: str, ssdp_headers: CaseInsensitiveDict) -> None:
        """Reinitialize device."""
        assert self.dmr_device is not None

        self.logger.debug("Reinitializing device, location: %s", location)

        new_device = await self._upnp_factory.async_create_device(location)
        self.dmr_device.profile_device.reinit(new_device)
        self.dmr_device.profile_device.ssdp_headers = ssdp_headers

    async def _create_device(self, location: str) -> None:
        # Connect to the base UPNP device
        upnp_device = await self._upnp_factory.async_create_device(location)

        # Check if the device is a DMR-profile device.
        # We are doing this here and this late to avoid fetching and parsing the
        # description.xml twice.
        if not DmrDevice.is_profile_device(upnp_device):
            self.logger.warning(
                "Found incompatible device, removing: %s",
                upnp_device.device_type,
            )
            self.provider.mass.players.remove(self.id)
            return

        # Create profile wrapper
        self.dmr_device = DmrDevice(upnp_device, self._event_handler)

    def _update_attributes(self) -> None:
        """Update attributes of the MA Player from DLNA state."""
        assert self.dmr_device is not None

        # generic attributes
        if self.available:
            self._player.available = True
            self._player.name = self.dmr_device.name
            self._player.volume_level = int((self.dmr_device.volume_level or 0) * 100)
            self._player.volume_muted = self.dmr_device.is_volume_muted or False
            self._player.state = self.get_state(self.dmr_device)
            self._player.current_item_id = self.dmr_device.current_track_uri or ""

            if self._player.player_id in self._player.current_item_id:
                self._player.active_source = self._player.player_id
            elif "spotify" in self._player.current_item_id:
                self._player.active_source = "spotify"
            elif self._player.current_item_id.startswith("http"):
                self._player.active_source = "http"
            else:
                # TODO: handle other possible sources here
                self._player.active_source = None

            if self.dmr_device.media_position:
                # only update elapsed_time if the device actually reports it
                self._player.elapsed_time = float(self.dmr_device.media_position)

                if self.dmr_device.media_position_updated_at is not None:
                    self._player.elapsed_time_last_updated = (
                        self.dmr_device.media_position_updated_at.timestamp()
                    )
        else:
            # device is unavailable
            self._player.available = False

    async def _update_player(self) -> None:
        """Update DLNA Player."""
        prev_url = self._player.current_item_id
        prev_state = self._player.state
        self._update_attributes()
        current_url = self._player.current_item_id
        current_state = self._player.state

        if (prev_url != current_url) or (prev_state != current_state):
            # fetch track details on state or url change
            self.force_poll = True

        # let the MA player manager work out if something actually updated
        self.provider.mass.players.update(self.id)

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
