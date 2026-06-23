# -*- coding: utf-8 -*-
import glob
import os
import sys
import tempfile
import atexit

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.report_ingestor.core import (  # noqa: E402
    _classify_sheet_type,
    _infer_header_metadata,
    clean_bs,
    clean_cf,
    clean_pl,
    load_config,
)


def _pick_sample() -> str:
    candidates = []
    for pattern in ("*月报*.xlsx", "报表*.xlsx", "*.xlsx"):
        candidates.extend(glob.glob(os.path.join(ROOT, pattern)))
    candidates = [p for p in candidates if os.path.isfile(p)]
    candidates.sort(key=lambda p: (os.path.basename(p).lower(), p.lower()))
    if not candidates:
        return _make_temp_sample()
    return candidates[0]


def _make_temp_sample() -> str:
    tmp = tempfile.NamedTemporaryFile(prefix="report_ingestor_sample_", suffix=".xlsx", delete=False)
    tmp.close()
    atexit.register(lambda p=tmp.name: os.path.exists(p) and os.remove(p))
    bs = pd.DataFrame(
        [
            ["单体资产负债表", None, None, None, None, None],
            ["2026年1月31日", None, None, "编制单位:测试有限公司", None, None],
            ["项目", "期末余额", "年初余额", "项目", "期末余额", "年初余额"],
            ["货币资金", 100, 80, "短期借款", 50, 40],
            ["资产总计", 100, 80, "负债和所有者权益总计", 50, 40],
        ]
    )
    pl = pd.DataFrame(
        [
            ["单体利润表", None, None],
            ["2026年1月31日", "编制单位:测试有限公司", None],
            ["项目", "本期金额", "本年累计金额"],
            ["营业收入", 120, 120],
            ["净利润", 20, 20],
        ]
    )
    cf = pd.DataFrame(
        [
            ["单体现金流量表", None, None],
            ["2026年1月31日", "编制单位:测试有限公司", None],
            ["项目", "本期金额", "本年累计金额"],
            ["销售商品、提供劳务收到的现金", 110, 110],
            ["经营活动产生的现金流量净额", 30, 30],
        ]
    )
    with pd.ExcelWriter(tmp.name) as writer:
        bs.to_excel(writer, sheet_name="资产负债表", index=False, header=False)
        pl.to_excel(writer, sheet_name="利润表", index=False, header=False)
        cf.to_excel(writer, sheet_name="现金流量表", index=False, header=False)
    return tmp.name


def _first_sheet_by_type(excel_file: pd.ExcelFile, sheet_type: str) -> str:
    for sheet_name in excel_file.sheet_names:
        if _classify_sheet_type(sheet_name) == sheet_type:
            return sheet_name
    raise AssertionError(f"未找到 {sheet_type} Sheet: {excel_file.sheet_names}")


def _assert_meta_columns(df: pd.DataFrame) -> None:
    assert not df.empty
    assert "主体" in df.columns
    assert "报表口径" in df.columns


def main() -> int:
    cfg = load_config(os.path.join(ROOT, "tools", "report_ingestor", "config.json"))
    sample = _pick_sample()
    print("Sample:", sample)

    excel_file = pd.ExcelFile(sample)
    try:
        print("Sheets:", excel_file.sheet_names)

        assert _classify_sheet_type("2603资产负债表") == "BS"
        assert _classify_sheet_type("母公司利润表") == "PL"
        assert _classify_sheet_type("合并现金流量表") == "CF"

        bs_sheet = _first_sheet_by_type(excel_file, "BS")
        pl_sheet = _first_sheet_by_type(excel_file, "PL")
        cf_sheet = _first_sheet_by_type(excel_file, "CF")

        meta = _infer_header_metadata(excel_file.parse(bs_sheet, header=None))
        print("Metadata:", meta)
        assert "主体" in meta
        assert "报表口径" in meta

        no_caliber_df = pd.DataFrame([["资产负债表", None], ["编制单位:测试有限公司", None], ["项目", "期末余额"]])
        assert _infer_header_metadata(no_caliber_df)["报表口径"] == ""

        df_bs = clean_bs(sample, bs_sheet, cfg, excel_file=excel_file)
        df_pl = clean_pl(sample, pl_sheet, cfg, excel_file=excel_file)
        df_cf = clean_cf(sample, cf_sheet, cfg, excel_file=excel_file)
        for df in (df_bs, df_pl, df_cf):
            _assert_meta_columns(df)
    finally:
        excel_file.close()

    print(f"Rows: BS={len(df_bs)} PL={len(df_pl)} CF={len(df_cf)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
