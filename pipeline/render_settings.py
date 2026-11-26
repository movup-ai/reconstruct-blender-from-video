"""Render quality selection independent of fitted character geometry."""


def engine(config=None, data=None):
    # The explicit LZHD delivery specification requires scene lighting.
    if (data or {}).get('spec')=='lzhd1':return 'eevee'
    value=(config or {}).get('render_engine','workbench')
    if value not in ('workbench','eevee'):raise ValueError('render_engine must be workbench or eevee')
    return value
