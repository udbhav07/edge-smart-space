"""A certificate of this node's own, so phones may use their microphone (FR-94).

Browsers allow a page the microphone only over HTTPS, or on localhost. A
phone reaching the console at ``http://192.168.x.y`` therefore cannot record
at all. The console also serves HTTPS, with a certificate made here, once,
for the names and addresses this machine answers to. Nothing vouches for it
but this node, so each device shows a warning the first time and the person
accepts it; after that it is an ordinary secure page.

Made again whenever the machine's addresses change, so it never names an
address it no longer has.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import socket
from pathlib import Path

LOGGER = logging.getLogger(__name__)

CERT_FILE = "console.crt"
KEY_FILE = "console.key"
#: Browsers refuse a server certificate valid for more than about two years.
_VALID_DAYS = 800


def local_names() -> tuple[set[str], set[str]]:
    """This machine's names and IPv4 addresses, loopback included."""
    names = {"localhost", socket.gethostname()}
    addresses = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError as exc:
        LOGGER.info("could not list this machine's addresses: %s", exc)
    return names, addresses


def ensure_certificate(directory: Path, names: set[str], addresses: set[str], now: dt.datetime) -> tuple[Path, Path]:
    """The certificate and key, made if missing, expiring, or not covering every address.

    :returns: paths to the certificate and the private key, PEM.
    """
    directory.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = directory / CERT_FILE, directory / KEY_FILE
    if cert_path.exists() and key_path.exists() and _still_good(cert_path, names, addresses, now):
        return cert_path, key_path
    _make(cert_path, key_path, names, addresses, now)
    LOGGER.info("made a certificate for %s", ", ".join(sorted(names | addresses)))
    return cert_path, key_path


def _still_good(cert_path: Path, names: set[str], addresses: set[str], now: dt.datetime) -> bool:
    from cryptography import x509

    try:
        certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
        covered = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except (ValueError, x509.ExtensionNotFound):
        return False
    have_names = set(covered.get_values_for_type(x509.DNSName))
    have_addresses = {str(address) for address in covered.get_values_for_type(x509.IPAddress)}
    fresh = certificate.not_valid_after_utc - now > dt.timedelta(days=30)
    return fresh and names <= have_names and addresses <= have_addresses


def _make(cert_path: Path, key_path: Path, names: set[str], addresses: set[str], now: dt.datetime) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Smart Space console")])
    alternatives = [x509.DNSName(name) for name in sorted(names)] + [
        x509.IPAddress(ipaddress.ip_address(address)) for address in sorted(addresses)
    ]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=_VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(alternatives), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_temporary = key_path.with_suffix(".tmp")
    key_temporary.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    key_temporary.replace(key_path)
    cert_temporary = cert_path.with_suffix(".tmp")
    cert_temporary.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    cert_temporary.replace(cert_path)
