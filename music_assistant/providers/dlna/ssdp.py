"""The SSDP discovery."""

from __future__ import annotations

import logging
from asyncio.events import AbstractEventLoop
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from async_upnp_client.aiohttp import AiohttpSessionRequester
from async_upnp_client.const import SsdpSource
from async_upnp_client.description_cache import DescriptionCache
from async_upnp_client.profiles.dlna import DmrDevice
from async_upnp_client.search import SsdpSearchListener
from async_upnp_client.ssdp import SSDP_MX, AddressTupleVXType
from async_upnp_client.ssdp_listener import SsdpDevice, SsdpDeviceTracker
from async_upnp_client.utils import CaseInsensitiveDict

if TYPE_CHECKING:
    from aiohttp.client import ClientSession

SSDP_ST_DMR = "urn:schemas-upnp-org:device:MediaRenderer:1"

# Attributes for accessing info from retrieved UPnP device description
ATTR_ST: Final = "st"
ATTR_NT: Final = "nt"
ATTR_UPNP_DEVICE_TYPE: Final = "deviceType"
ATTR_UPNP_FRIENDLY_NAME: Final = "friendlyName"
ATTR_UPNP_MANUFACTURER: Final = "manufacturer"
ATTR_UPNP_MANUFACTURER_URL: Final = "manufacturerURL"
ATTR_UPNP_MODEL_DESCRIPTION: Final = "modelDescription"
ATTR_UPNP_MODEL_NAME: Final = "modelName"
ATTR_UPNP_MODEL_NUMBER: Final = "modelNumber"
ATTR_UPNP_MODEL_URL: Final = "modelURL"
ATTR_UPNP_SERIAL: Final = "serialNumber"
ATTR_UPNP_SERVICE_LIST: Final = "serviceList"
ATTR_UPNP_UDN: Final = "UDN"
ATTR_UPNP_UPC: Final = "UPC"
ATTR_UPNP_PRESENTATION_URL: Final = "presentationURL"


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SsdpServiceInfo:
    """Prepared info from ssdp/upnp entries."""

    ssdp_usn: str
    ssdp_st: str
    ssdp_location: str
    ssdp_udn: str
    upnp: Mapping[str, Any]
    ssdp_nt: str | None = None
    ssdp_ext: str | None = None
    ssdp_server: str | None = None
    ssdp_headers: Mapping[str, Any] = field(default_factory=dict)
    ssdp_all_locations: set[str] = field(default_factory=set)


DiscoverCallback = Callable[[SsdpServiceInfo, SsdpSource], Coroutine[Any, Any, None]]


class SSDPScanner:
    """Class to manage SSDP searching and SSDP advertisements."""

    def __init__(
        self,
        session: ClientSession,
        on_discover: DiscoverCallback,
        loop: AbstractEventLoop | None = None,
        source: AddressTupleVXType | None = None,
        target: AddressTupleVXType | None = None,
        timeout: int = SSDP_MX,
        search_target: str = SSDP_ST_DMR,
    ) -> None:
        """Initialize class."""
        self._listener = SsdpSearchListener(
            async_callback=self._on_discovery,
            loop=loop,
            source=source,
            target=target,
            timeout=timeout,
            search_target=search_target,
            async_connect_callback=self._async_connect_callback,
        )
        self._device_tracker = SsdpDeviceTracker()
        self._description_cache: DescriptionCache | None = None
        self._session = session
        self._discover_callback = on_discover

    @property
    def devices(self) -> list[SsdpDevice]:
        """Get all seen devices."""
        return list(self._device_tracker.devices.values())

    async def async_start(self) -> None:
        """Start the listener."""
        requester = AiohttpSessionRequester(self._session, True, 10)
        self._description_cache = DescriptionCache(requester)

        await self._listener.async_start()

    async def async_stop(self) -> None:
        """Stop the listener."""
        self._listener.async_stop()

    async def async_scan(self, *_: Any) -> None:
        """Scan for new entries using ssdp listeners."""
        await self._listener.async_start()

    async def _async_connect_callback(self) -> None:
        self._listener.async_search()

    async def _on_discovery(self, headers: CaseInsensitiveDict) -> None:
        """Combine the headers and description into discovery_info."""
        (
            propagate,
            ssdp_device,
            device_or_service_type,
            ssdp_source,
        ) = self._device_tracker.see_search(headers)

        if propagate and ssdp_device and device_or_service_type:
            assert ssdp_source is not None

            info_desc = await self._async_get_description_dict(headers["location"])

            discovery_info = discovery_info_from_headers_and_description(
                ssdp_device, headers, info_desc
            )

            if not _is_dmr_device(discovery_info) or _is_sonos_device(discovery_info):
                return

            await self._discover_callback(discovery_info, ssdp_source)

    async def _async_get_description_dict(self, location: str | None) -> Mapping[str, str]:
        """Get description dict."""
        assert self._description_cache is not None
        cache = self._description_cache

        has_description, description = cache.peek_description_dict(location)
        if has_description:
            return description or {}

        return await cache.async_get_description_dict(location) or {}


def discovery_info_from_headers_and_description(
    ssdp_device: SsdpDevice,
    combined_headers: CaseInsensitiveDict,
    info_desc: Mapping[str, Any],
) -> SsdpServiceInfo:
    """Convert headers and description to discovery_info."""
    ssdp_usn = combined_headers["usn"]
    ssdp_st = combined_headers.get_lower("st")
    if isinstance(info_desc, CaseInsensitiveDict):
        upnp_info = {**info_desc.as_dict()}
    else:
        upnp_info = {**info_desc}

    # Increase compatibility: depending on the message type,
    # either the ST (Search Target, from M-SEARCH messages)
    # or NT (Notification Type, from NOTIFY messages) header is mandatory
    if not ssdp_st:
        ssdp_st = combined_headers["nt"]

    # Ensure UPnP "udn" is set
    if ATTR_UPNP_UDN not in upnp_info:
        if udn := _udn_from_usn(ssdp_usn):
            upnp_info[ATTR_UPNP_UDN] = udn

    return SsdpServiceInfo(
        ssdp_usn=ssdp_usn,
        ssdp_st=ssdp_st,
        ssdp_ext=combined_headers.get_lower("ext"),
        ssdp_server=combined_headers.get_lower("server"),
        ssdp_location=combined_headers.get_lower("location"),
        ssdp_udn=combined_headers.get_lower("_udn"),
        ssdp_nt=combined_headers.get_lower("nt"),
        ssdp_headers=combined_headers,
        upnp=upnp_info,
        ssdp_all_locations=set(ssdp_device.locations),
    )


def _udn_from_usn(usn: str | None) -> str | None:
    """Get the UDN from the USN."""
    if usn is None:
        return None

    if usn.startswith("uuid:"):
        return usn.split("::")[0]

    return None


def _is_sonos_device(discovery_info: SsdpServiceInfo) -> bool:
    return "rincon" in discovery_info.ssdp_udn


def _is_dmr_device(discovery_info: SsdpServiceInfo) -> bool:
    """Determine if discovery is a complete DLNA DMR device.

    Use the discovery_info instead of DmrDevice.is_profile_device to avoid
    contacting the device again.
    """
    # Abort if the device doesn't support all services required for a DmrDevice.
    discovery_service_list = discovery_info.upnp.get(ATTR_UPNP_SERVICE_LIST)
    if not discovery_service_list:
        return False

    services = discovery_service_list.get("service")
    if not services:
        discovery_service_ids: set[str] = set()
    elif isinstance(services, list):
        discovery_service_ids = {service.get("serviceId") for service in services}
    else:
        # Only one service defined (etree_to_dict failed to make a list)
        discovery_service_ids = {services.get("serviceId")}

    return DmrDevice.SERVICE_IDS.issubset(discovery_service_ids)


def get_preferred_location(locations: set[str]) -> str:
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
