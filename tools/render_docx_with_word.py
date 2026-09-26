from __future__ import annotations

import argparse
from pathlib import Path

import pythoncom
import win32com.client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    print("WORD_INSTANCE_OK", flush=True)
    doc = None
    try:
        doc = word.Documents.Open(
            str(args.docx.resolve()),
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
            Revert=False,
            OpenAndRepair=True,
            NoEncodingDialog=True,
        )
        print("WORD_OPEN_OK", flush=True)
        pages = doc.ComputeStatistics(2)
        doc.ExportAsFixedFormat(str(args.pdf.resolve()), 17)
        print(f"WORD_EXPORT_OK pages={pages}", flush=True)
    finally:
        if doc is not None:
            doc.Close(False)
        word.Quit()
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    main()
