from fastapi.routing import APIWebSocketRoute

from backend.app import app


def test_single_stream_websocket_route_is_registered_from_streams_router():
    routes = [
        route
        for route in app.routes
        if isinstance(route, APIWebSocketRoute) and route.path == "/streams/{session_id}"
    ]

    assert len(routes) == 1
    assert routes[0].endpoint.__module__ == "backend.routers.streams"