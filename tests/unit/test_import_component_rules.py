"""Exercise the maintainer's Excel-to-JSON workflow with both table layouts."""

import pytest
from openpyxl import Workbook

from app.core.component_rules import load_component_rules
from app.tools.import_component_rules import LIMIT_HEADERS, RANGE_HEADERS, main


def test_import_normalizes_percent_cells_and_named_sheet(tmp_path):
    source, output = tmp_path / "source.xlsx", tmp_path / "rules.json"
    workbook = Workbook()
    sheet = workbook.create_sheet("Rules")
    sheet.append(LIMIT_HEADERS)
    sheet.append(("U0500", "啟用", 0.01, 1.8, 0.1, 1.6))
    sheet.append(("others", "啟用", 0.3, 1.7, 0.5, 1.5))
    for row in sheet.iter_rows(min_row=2, min_col=3):
        for cell in row:
            cell.number_format = "0%"
    workbook.save(source)

    assert main([str(source), str(output), "--sheet", "Rules"]) == 0
    rules = load_component_rules(output)
    assert rules.components["U0500"].volume.model_dump() == {
        "min_pct": 1,
        "max_pct": 180,
    }
    assert rules.others.area.model_dump() == {"min_pct": 50, "max_pct": 150}


def test_import_rejects_duplicate_before_overwriting_valid_rules(tmp_path, capsys):
    source, output = tmp_path / "source.xlsx", tmp_path / "rules.json"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(RANGE_HEADERS)
    sheet.append(("others", "啟用", "30% ~ 170%", "50% ~ 150%"))
    workbook.save(source)
    assert main([str(source), str(output)]) == 0
    before = output.read_bytes()

    sheet.append((" others ", "啟用", "30% ~ 180%", "60% ~ 160%"))
    workbook.save(source)
    with pytest.raises(SystemExit) as exc:
        main([str(source), str(output)])
    assert exc.value.code == 2
    assert "Duplicate component" in capsys.readouterr().err
    assert output.read_bytes() == before
