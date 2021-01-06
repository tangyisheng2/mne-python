from datetime import datetime


def convert_event_datetime(datetime_str: str):
    """

    :param datetime_str: 日期数组
    """
    return datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f+08:00')


if __name__ == '__main__':
    print(convert_event_datetime("2021-01-03T19:20:37.899215+08:00"))
