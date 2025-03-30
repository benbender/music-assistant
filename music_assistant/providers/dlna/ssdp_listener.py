"""async_upnp_client.ssdp_listener module."""

from asyncio.events import AbstractEventLoop
from collections.abc import Callable, Coroutine, KeysView, Mapping  # pylint: disable=import-error
from typing import Any

from aiohttp.client import ClientSession
from async_upnp_client.advertisement import SsdpAdvertisementListener
from async_upnp_client.aiohttp import AiohttpSessionRequester
from async_upnp_client.const import (
    AddressTupleVXType,
    DeviceOrServiceType,
    SsdpSource,
)
from async_upnp_client.description_cache import DescriptionCache
from async_upnp_client.profiles.dlna import DmrDevice
from async_upnp_client.search import SsdpSearchListener
from async_upnp_client.ssdp import SSDP_MX, SSDP_ST_ALL
from async_upnp_client.ssdp_listener import (
    SsdpDevice,
    SsdpDeviceTracker,
)
from async_upnp_client.ssdp_listener import (
    SsdpListener as BaseSSDPListener,
)

ATTR_SSDP_BOOTID = "BOOTID.UPNP.ORG"
ATTR_SSDP_NEXTBOOTID = "NEXTBOOTID.UPNP.ORG"
ATTR_SSDP_CONFIGID = "CONFIGID.UPNP.ORG"
ATTR_SSDP_LOCATION = "ssdp_location"

SSDPEventCallback = Callable[
    [SsdpDevice, DeviceOrServiceType, SsdpSource], Coroutine[Any, Any, None]
]


class SSDPListener(BaseSSDPListener):
    """SSDP Search and Advertisement listener."""

    def __init__(
        self,
        async_callback: SSDPEventCallback,
        session: ClientSession,
        source: AddressTupleVXType | None = None,
        target: AddressTupleVXType | None = None,
        loop: AbstractEventLoop | None = None,
        search_timeout: int = SSDP_MX,
        search_target: str = SSDP_ST_ALL,
        profile_service_ids: frozenset[str] = DmrDevice.SERVICE_IDS,
    ) -> None:
        """Initialize the listener."""
        self._search_target = search_target
        self._description_cache: DescriptionCache | None = None
        self._session = session
        self._profile_service_ids = profile_service_ids

        super().__init__(
            async_callback=async_callback,
            source=source,
            target=target,
            loop=loop,
            search_timeout=search_timeout,
            device_tracker=SsdpDeviceTracker(),
        )

    async def async_start(self) -> None:
        """Start search listener/advertisement listener."""
        self._description_cache = DescriptionCache(AiohttpSessionRequester(self._session, True, 10))

        self._advertisement_listener = SsdpAdvertisementListener(
            on_alive=self._on_alive,
            on_update=self._on_update,
            on_byebye=self._on_byebye,
            source=self.source,
            target=self.target,
            loop=self.loop,
        )
        await self._advertisement_listener.async_start()

        self._search_listener = SsdpSearchListener(
            callback=self._on_search,
            loop=self.loop,
            source=self.source,
            target=self.target,
            timeout=self.search_timeout,
            search_target=self._search_target,
        )
        await self._search_listener.async_start()

    async def _async_get_description_dict(self, location: str | None) -> Mapping[str, str]:
        """Get description dict."""
        assert self._description_cache is not None
        cache = self._description_cache

        has_description, description = cache.peek_description_dict(location)
        if has_description:
            return description or {}

        return await cache.async_get_description_dict(location) or {}


def get_preferred_location(locations: KeysView[str]) -> str:
    """Get the preferred location (an IPv4 location) from a set of locations."""
    # Prefer IPv4 over IPv6.
    for location in locations:
        if location.startswith(("http://[", "https://[")):
            continue

        return location

    # Fallback to any.
    for location in locations:
        return location

    raise ValueError("No location found")
