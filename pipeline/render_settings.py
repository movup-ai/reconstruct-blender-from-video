"""Render quality selection independent of fitted character geometry: Workbench by default, Eevee only when
config.json says render_engine = eevee."""


def engine(config=None, data=None):
    value=(config or {}).get('render_engine','workbench')
    if value not in ('workbench','eevee'):raise ValueError('render_engine must be workbench or eevee')
    return value
