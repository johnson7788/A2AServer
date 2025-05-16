#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/4/29 (Modified: 2025-04-22)
# @File  : form_tool.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : eg: 搜索工具
import time
import json
from fastmcp import FastMCP
from typing import Any, Optional
import random

mcp = FastMCP("搜索工具")

@mcp.tool()
def create_request_form(
    date: Optional[str] = None,
    amount: Optional[str] = None,
    purpose: Optional[str] = None,
) -> dict[str, Any]:
    """为员工创建一个报销发票的表单。
    参数:
        date (str): 请求日期。可以为空字符串。
        amount (str): 请求金额。可以为空字符串。
        purpose (str): 请求目的。可以为空字符串。
    返回:
        dict[str, Any]: 包含请求表单数据的字典。
    """
    request_id = 'request_id_' + str(random.randint(1000000, 9999999))
    return {
        'request_id': request_id,
        'date': '<transaction date>' if not date else date,
        'amount': '<transaction dollar amount>' if not amount else amount,
        'purpose': '<business justification/purpose of the transaction>'
        if not purpose
        else purpose,
    }

@mcp.tool()
def return_voice_form(
    form_request: dict[str, Any]) -> dict[str, Any]:
    """
    返回一个报销需要的表单。
    参数:
        form_request (dict[str, Any]): 请求表单数据。
    返回:
        dict[str, Any]: 表单响应的JSON字典。
    """
    if isinstance(form_request, str):
        form_request = json.loads(form_request)

    form_dict = {
        'type': 'form',
        'form': {
            'type': 'object',
            'properties': {
                'date': {
                    'type': 'string',
                    'format': 'date',
                    'description': 'Date of expense',
                    'title': 'Date',
                },
                'amount': {
                    'type': 'string',
                    'format': 'number',
                    'description': 'Amount of expense',
                    'title': 'Amount',
                },
                'purpose': {
                    'type': 'string',
                    'description': 'Purpose of expense',
                    'title': 'Purpose',
                },
                'request_id': {
                    'type': 'string',
                    'description': 'Request id',
                    'title': 'Request ID',
                },
            },
            'required': list(form_request.keys()),
        },
        'form_data': form_request
    }
    return json.dumps(form_dict)

@mcp.tool()
def submit_form(form_response: dict[str, Any]) -> str:
    """提交一个表单。
    参数:
        form_response (dict[str, Any]): 表单响应。
    返回:
        str: 提交表单的结果。
    """
    if isinstance(form_response, str):
        form_response = json.loads(form_response)
    print(f"表单已经提交成功了, {form_response}")
    return 'Form submitted successfully'

if __name__ == '__main__':
    result = create_request_form(date='2025-05-01',amount='10',purpose='午饭')
    print(result)
    result = return_voice_form(form_request={'amount': '10', 'date': '2025-05-01', 'purpose': '午饭', 'request_id': 'request_id_7016809'})
    print(result)
    result = submit_form(form_response={'amount': '10', 'date': '2025-05-01', 'purpose': '午饭', 'request_id': 'request_id_7016809'})
    print(result)