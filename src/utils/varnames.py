from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ColNames:
    SUPPLIER: str = 'Marca'
    PLATFORM: str = 'Plataforma'
    SKU: str = 'sku'
    DESCRIPTION: str = 'Descripción'
    LICENSE: str = 'Categoría'
    MONTH: str = 'Mes'
    YEAR: str = 'Año'
    UNITS: str = 'piezas'
    SALES_MXN: str = 'Precio venta'
    COST: str = 'Costo Alpha'
    DATE: str = 'date'
    MONTH_NUM: int = 'month_num'
    PRICE: float = 'price'
    PLATFORM_MONTHLY_SALES: float = 'platform_monthly_sales'
    SKU_PLATFORM: str = 'sku_platform'
    PRODUCT: str = 'product'
    CLUSTER: int = 'cluster'
    STOCKOUT: bool = 'stockout'



MONTH_NAME_TO_NUMBER = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

CATEGORIES: list[str | Any] = ['organizador', 'taza', 'termo', 'mochila', 'lonchera', 'lampara', 'gorro', 'gorra', 'almacenamiento', 'iman',
          'perchero', 'display', 'tuper', 'repisa', 'despertador', 'cartera', 'guante', 'calcetas', 'beanie', 'infusor',
          'alcancia', 'pelotas antiestres', 'llavero', 'desk mat', 'libreta', 'botella', 'vaso',
          'figuras coleccionables', 'juego de mesa', 'stickers', 'mini mochila', 'mochila con ruedas', 'lapicera',
          'rompecabezas', 'plato', 'reloj', 'soporte para telefono', 'luz', 'escritorio', 'cabeza', 'brick', 'bolsa',
          'marco', 'bloque', 'set magnetico', 'brazalete', 'black', 'estuche', 'libr', 'funda', 'cap', 'correa',
          'bolso', 'backpack', 'soporte', 'estante', 'pin', 'pelota', 'case', 'minibackpack', 'moch', 'pulsera',
          'control', 'retratos', 'portalapices', 'light', 'clasificadora', 'p. rev', 'peluche', 'vela', 'rack']

CATEGORIES_DICT = {"beanie": "guante",
               "bloque": "brick",
               "libr": "librero",
               "backpack": "mochila",
               "moch": "mochila",
               "retratos": "marco",
               "estante": "librero",
               "clasificadora": "organizador",
               "p. rev": "peluche",
               "rack": "organizador"}