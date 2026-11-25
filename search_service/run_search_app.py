# -*- coding: utf-8 -*-
"""启动对话检索 + PDF 预览服务。"""

from settings import bootstrap, get_api_config

if __name__ == "__main__":
    import uvicorn

    bootstrap()
    api = get_api_config()
    uvicorn.run("search_app:app", host=api.host, port=api.port, reload=False)
