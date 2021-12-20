def timefeature(dates):
    dates["hour"] = dates["date"].apply(lambda row: row.hour / 23 - 0.5, 1)  # 一天中的第几小时
    dates["weekday"] = dates["date"].apply(lambda row: row.weekday() / 6 - 0.5, 1)  # 周几
    dates["day"] = dates["date"].apply(lambda row: row.day / 30 - 0.5, 1)  # 一个月的第几天
    dates["month"] = dates["date"].apply(lambda row: row.month / 365 - 0.5, 1)  # 一年的第几天
    return dates[["hour", "weekday", "day", "month"]].values
