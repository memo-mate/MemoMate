from openpyxl import load_workbook


def parse_excel(file_path):
    workbook = load_workbook(file_path)
    sheet = workbook.active
    return sheet
