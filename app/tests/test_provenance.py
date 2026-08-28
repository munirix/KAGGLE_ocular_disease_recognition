"""test_provenance.py -- manifesto de execucao e metadados de reprodutibilidade.

Garante que o manifesto registre as dependencias de modelagem (incluindo mord,
que sustenta a regressao ordinal da Etapa 3) e o consumo de memoria de pico,
evitando que uma lacuna de proveniencia passe silenciosamente.
"""
from __future__ import annotations

from src import config as C
from src.provenance import (build_manifest, file_sha256, get_environment_info,
                            peak_memory_mb)


def test_environment_lists_modelling_dependencies():
    versions = get_environment_info()["package_versions"]
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "mord"):
        assert pkg in versions, f"{pkg} ausente do manifesto"
        assert versions[pkg] != "not installed", f"{pkg} declarado mas nao instalado"


def test_environment_records_platform_and_commit():
    info = get_environment_info()
    assert info["python_version"]
    assert info["platform"]
    assert "git_commit" in info


def test_peak_memory_is_positive_or_unavailable():
    value = peak_memory_mb()
    assert value is None or value > 0


def test_manifest_contains_config_cache_and_memory():
    manifest = build_manifest(C.RunConfig(), command="teste")
    assert manifest["command"] == "teste"
    assert "config" in manifest
    assert "seed" in manifest["config"]
    assert "cache" in manifest
    assert "features_cache" in manifest["cache"]
    assert "peak_memory_mb" in manifest


def test_manifest_merges_extra_fields():
    manifest = build_manifest(C.RunConfig(), command="stage2",
                              extra={"stage": "stage2", "n_features": 169})
    assert manifest["stage"] == "stage2"
    assert manifest["n_features"] == 169


def test_file_sha256_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("conteudo", encoding="utf-8")
    assert file_sha256(a) == file_sha256(a)

    b = tmp_path / "b.txt"
    b.write_text("conteudo diferente", encoding="utf-8")
    assert file_sha256(a) != file_sha256(b)