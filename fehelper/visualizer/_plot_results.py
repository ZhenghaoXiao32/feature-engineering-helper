import plotly.graph_objects as go


def plot_results(result_dfs, df_ids, result='accuracy'):
    """
    Plot the results dataframe.

    :param result_dfs: the list of test result dataframe
    :param df_ids: the list of different test result ids
    :param result: "accuracy" or "f1_score"
    :return:
    """
    if result == 'accuracy':
        fig = go.Figure()
        for df, df_id in zip(result_dfs, df_ids):
            fig.add_trace(go.Scatter(x=df.model, y=df.test_acc,
                                     mode='lines+markers', name=df_id))
        fig.update_layout(title={'text': 'Testing Accuracy of Models',
                                     'y': 0.9,
                                     'x': 0.4,
                                     'xanchor': 'center',
                                     'yanchor': 'top'})
    if result == 'f1_score':
        fig = go.Figure()
        for df, df_id in zip(result_dfs, df_ids):
            fig.add_trace(go.Scatter(x=df.model, y=df.f1_score,
                                     mode='lines+markers', name=df_id))
        fig.update_layout(title={'text': 'Testing F1 Score of Models',
                                 'y': 0.9,
                                 'x': 0.4,
                                 'xanchor': 'center',
                                 'yanchor': 'top'})

    fig.show()
