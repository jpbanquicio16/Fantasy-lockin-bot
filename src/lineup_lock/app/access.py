"""The supported deployment is localhost or a Tailscale interface, never a public bind."""
import ipaddress


def validate_bind_address(address: str) -> None:
    if address == "localhost":
        return
    ip = ipaddress.ip_address(address)
    tailscale = (ip in ipaddress.ip_network("100.64.0.0/10") if ip.version == 4
                 else ip in ipaddress.ip_network("fd7a:115c:a1e0::/48"))
    if not ip.is_loopback and not tailscale:
        raise ValueError("Bind to 127.0.0.1 or this host's Tailscale IP; public/wildcard binds are disabled")
