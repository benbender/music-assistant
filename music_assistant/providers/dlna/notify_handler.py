"""Various helpers and utils for the DLNA Player Provider."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiohttp.web import Request, Response
from async_upnp_client.const import HttpRequest
from async_upnp_client.event_handler import UpnpEventHandler, UpnpNotifyServer

if TYPE_CHECKING:
    from async_upnp_client.client import UpnpRequester

    from music_assistant import MusicAssistant  # type: ignore[attr-defined]


_LOGGER = logging.getLogger(__name__)
_LOGGER_TRAFFIC_UPNP = logging.getLogger("async_upnp_client.traffic.upnp")


class DLNANotifyHandler(UpnpNotifyServer):
    """Notify handler for async_upnp_client which uses the MA webserver."""

    is_registered: bool = False

    def __init__(
        self,
        mass: MusicAssistant,
        requester: UpnpRequester,
    ) -> None:
        """Initialize notify handler."""
        self._mass = mass
        self.event_handler = UpnpEventHandler(self, requester)

    def register(self) -> None:
        """Register dynamic route."""
        _LOGGER.debug("notify handler registered")

        self._mass.streams.register_dynamic_route("/notify", self._handle_request, method="NOTIFY")
        self.is_registered = True

    def unregister(self) -> None:
        """Unregister dynamic route."""
        if self.is_registered:
            _LOGGER.debug("notify handler unregistered")

            self._mass.streams.unregister_dynamic_route("/notify", "NOTIFY")
            self.is_registered = False

    async def _handle_request(self, request: Request) -> Response:
        """Handle incoming requests."""
        assert self.is_registered is True

        _LOGGER.debug("Received request: %s", request)
        log_traffic = _LOGGER_TRAFFIC_UPNP.isEnabledFor(logging.DEBUG)

        headers = request.headers
        body = await request.text()
        if log_traffic:
            _LOGGER_TRAFFIC_UPNP.debug(
                "Incoming request:\nNOTIFY\n%s\n\n%s",
                "\n".join([key + ": " + value for key, value in headers.items()]),
                body,
            )

        if request.method != "NOTIFY":
            _LOGGER.debug("Not notify")

            return Response(status=405)

        # transform aiohttp request to async_upnp_client request
        http_request = HttpRequest(
            method=request.method,
            url=str(request.url),
            headers=headers,
            body=body,
        )

        status = await self.event_handler.handle_notify(http_request)

        _LOGGER.debug("NOTIFY response status: %s", status)

        if log_traffic:
            _LOGGER_TRAFFIC_UPNP.debug("Sending response: %s", status)

        return Response(status=status)

    @property
    def callback_url(self) -> str:
        """Return callback URL on which we are callable."""
        return f"{self._mass.streams.base_url}/notify"
