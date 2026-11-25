# -*- coding: utf-8 -*-
"""启动入库 / OA / 二开服务。"""

from settings import bootstrap, get_api_config

if __name__ == "__main__":
    import uvicorn

    bootstrap()
    api = get_api_config()
    uvicorn.run("ingest_app:app", host=api.host, port=api.port, reload=False)
