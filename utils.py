def order_data_by_million(table):
    """
    This function takes in a table and returns a
    sorted table by the number of million in descending order.
    """
    money_index = table.column_names.index("earning_($ million)")
    sorted_table = table.copy()
    for row in table.data:
        pass
    return sorted_table