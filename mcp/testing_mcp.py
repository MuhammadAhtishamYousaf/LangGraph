import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test_mcp_add_and_list():
    client = MultiServerMCPClient(
        {
            "expense": {
                "transport": "streamable_http",
                "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
            }
        }
    )

    # Fetch tools
    tools = await client.get_tools()

    add_expense = next(t for t in tools if t.name == "add_expense")
    list_expenses = next(t for t in tools if t.name == "list_expenses")

    # 1️⃣ Add an expense
    add_result = await add_expense.ainvoke(
        {
            "date": "2024-11-10",
            "amount": 500,
            "category": "Food",
            "subcategory": "Dinner",
            "note": "Pizza"
        }
    )

    print("\nAdd Expense Result:\n")
    print(add_result)

    # 2️⃣ List all expenses
    list_result = await list_expenses.ainvoke(
        {
            "start_date": "2024-11-01",
            "end_date": "2024-11-30",
        }
    )

    print("\nAll Expenses:\n")
    print(list_result)

if __name__ == "__main__":
    asyncio.run(test_mcp_add_and_list())
