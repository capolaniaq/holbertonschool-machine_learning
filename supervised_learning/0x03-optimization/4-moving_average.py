#!/usr/bin/env python3
"""
calculates the weighted moving average of a data set
"""


def moving_average(data, beta):
    """
    moving_average - calculates the weighted moving average of a data set
    :param data: list of data points
    :param beta: weight of the moving average
    :return: list of weighted moving averages
    """
    if beta < 0 or beta > 1:
        return None
    m_avg = []
    for i in range(len(data)):
        if i == 0:
            m_avg.append(data[i])
        else:
            m_avg.append(beta * m_avg[i - 1] + (1 - beta) * data[i])
    return m_avg
