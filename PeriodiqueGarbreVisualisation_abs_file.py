import sys

import dash
import dash_cytoscape as cyto
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Output, Input
import re


def other(tab, i, elements, k=0, u=0, nLeaf=0, selected=-1):
    tmp = []
    rx = re.search('\((.*)\)\[(.*)]', tab)
    r = rx[1].split(sep=' ')
    elements = elements + [{
        'data': {'id': str(i), 'label': rx[2]},
        'position': {'x': 100 * i, 'y': 50 * k},
    }, ]
    j = i
    i = i + 1
    k = i
    rm = 0
    for elem in range(0, len(r)):
        if rm > 0:
            rm = rm + len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp[len(tmp) - 1] = tmp[len(tmp) - 1] + ' ' + r[elem]
        elif r[elem][0] == 'p':
            tmp[len(tmp) - 1] = tmp[len(tmp) - 1] + ' ' + r[elem]
        elif r[elem][0] == '(':
            rm = len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp.append(r[elem])


        else:
            tmp.append(r[elem])

    for e in range(0, len(tmp)):
        if not tmp[e][0] == '(' and not tmp[e][0] == '[':
            y = i - 1
            if selected == nLeaf:
                elements = elements + [{
                    'data': {'id': str(i), 'label': tmp[e], 'leaf': nLeaf},
                    'position': {'x': 100 * y, 'y': 50 * k},
                    'style': {
                        'shape': 'square',
                        'color': px.colors.qualitative.Alphabet[nLeaf],
                        'background-color': px.colors.qualitative.Alphabet[nLeaf],
                    }
                },
                    {
                        'data': {'source': str(i), 'target': str(j)}
                    }
                ]
            else:
                elements = elements + [{
                    'data': {'id': str(i), 'label': tmp[e], 'leaf': nLeaf},
                    'position': {'x': 100 * y, 'y': 50 * k},
                    'style': {
                        'shape': 'square',
                        'background-color': px.colors.qualitative.Alphabet[nLeaf]
                    }
                },
                    {
                        'data': {'source': str(i), 'target': str(j)}
                    }
                ]
            nLeaf = nLeaf + 1
            i = i + 1

        if tmp[e][0] == '(':
            elements = elements + [{
                'data': {'source': str(i), 'target': str(j)}
            }
            ]
            t = other(tmp[e], i, elements, k, nLeaf=nLeaf, selected=selected)
            u = i
            elements = t[0]
            i = t[1]
            nLeaf = t[2]

        if e < len(tmp) - 1 and tmp[e + 1][1] == 'd':
            if tmp[e][0] == '(':
                elements = elements + [{
                    'data': {'source': str(u), 'target': str(i), 'label': tmp[e + 1]},
                    'classes': 'labelled'
                }]
            else:
                elements = elements + [{
                    'data': {'source': str(i - 1), 'target': str(i), 'label': tmp[e + 1]},
                    'classes': 'labelled'
                }]
    t = [elements]
    t.append(i)
    t.append(nLeaf)
    return t


# We create a dataFrame with id, label, period, rep, nested levels, deltas we don't care about shifts yet
def createDataFrame(tab, i, row, k=1, u=0):
    tmp = []
    rx = re.search('\((.*)\)\[r=(.*?) p=(.*)]', tab)
    r = rx[1].split(sep=' ')
    ## Create a row with p and r and nN = 1 and d = 0
    row[0].append('')
    row[1].append(rx[3])
    row[2].append(rx[2])
    row[3].append(k)
    row[4].append(0)
    # Do not change this for : we will use this format after for our things
    rm = 0
    for elem in range(0, len(r)):
        if rm > 0:
            rm = rm + len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp[len(tmp) - 1] = tmp[len(tmp) - 1] + ' ' + r[elem]
        elif r[elem][0] == 'p':
            tmp[len(tmp) - 1] = tmp[len(tmp) - 1] + ' ' + r[elem]
        elif r[elem][0] == '(':
            rm = len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp.append(r[elem])


        else:
            tmp.append(r[elem])
    # In this for :
    #   add the label at the right place (last row)
    #   add a row for each d =
    for e in range(0, len(tmp)):
        if not tmp[e][0] == '(' and not tmp[e][0] == '[':
            row[0][len(row[0]) - 1] = tmp[e]

        if tmp[e][0] == '(':
            k = k + 1
            row = createDataFrame(tmp[e], i, row, k, u)
            k = k - 1

        if tmp[e][1] == 'd':
            te = re.search('d=(.*)', tmp[e])
            row[0].append('')
            row[1].append(0)
            row[2].append(0)
            row[3].append(k)
            row[4].append(te[1])

    return row


def computeShiftsNested(row, tabE, tabt, tabY):
    t0 = re.search('=(\d+)', tabt)
    t0 = int(t0[1])
    lE = re.split(' ', tabE)
    # Tabs to create DataFrame we insert first elem later
    x = []
    y = []
    label = []
    ids = []
    size = []
    shift = []
    rep = []
    timeX = re.split('; ', tabY)
    timeX[0] = timeX[0][3:]
    timeS = re.split('; ', tabE)
    timeS[0] = timeS[0][7:]
    for elem in range(len(row[2])):
        if row[3][elem] == 1:
            for nRep in range(int(row[2][elem])):
                i = elem + 1

                if not row[0][elem] == '':
                    x.append(timeX[len(x)])
                    y.append(1 - row[3][elem] - nRep * 0.05)
                    ids.append(str(elem))
                    label.append(row[0][elem])
                    size.append(12)
                    shift.append(timeS[len(shift)])
                    rep.append(nRep)

                while i < len(row[2]):
                    if row[3][i] > row[3][elem]:
                        nLevels(row, t0, lE, x, y, label, ids, i, timeX[len(x)], timeX, shift, timeS, rep, nRep)

                    elif not row[0][i] == '':
                        x.append(timeX[len(x)])
                        y.append(1 - row[3][i] - nRep * 0.05)
                        ids.append(str(i))
                        label.append(row[0][i])
                        size.append(12)
                        shift.append(timeS[len(shift)])
                        rep.append(nRep)

                    i = i + 1

    while len(size) < len(x):
        size.append(12)
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "id": ids,
        "label": label,
        "size": size,
        "ideal": shift,
        "nRep": rep,
    })
    return df


def nLevels(row, t0, lE, x, y, label, ids, elem, tmpTime, timeX, shift, timeS, rep, occ):

    for nRep in range(int(row[2][elem])):
        i = elem + 1

        if not row[0][elem] == '':

            x.append(timeX[len(x)])
            y.append(1 - row[3][elem])
            ids.append(str(elem))
            label.append(row[0][elem])
            shift.append(timeS[len(shift)])
            rep.append(occ)

        while i < len(row[2]):

            if row[3][i] > row[3][elem]:

                nLevels(row, t0, lE, x, y, label, ids, i, tmpTime, timeX, shift, timeS, rep, occ)

            i = i + 1


app = dash.Dash(__name__)

def pre_visu(patternId):
    t = line[patternId]
    tab = t.split(sep="\t")
    t0 = 0
    p = 0
    p0= 0
    dE =''
    tXstar = ''
    tX = ''
    for elem in tab:
        if elem[0:2] == "t0":
            t0 = elem
            continue
        if elem[0] == "(":
            content = elem
            p = re.split(' ', elem)
            p0 = re.search('p=(.*)]', p[len(p) - 1])
        if elem[0:6] == "tXstar":
            tXstar = elem
            dE = re.split(' ', elem)
        if elem[0:3] == "tX=":
            tX = elem
    # For TimeLine Construction
    i = 0
    row = createDataFrame(content, i, [[], [], [], [], []])
    df = computeShiftsNested(row, tXstar, t0, tX)

    return content, t0, dE, row, df, i


@app.callback(
    Output('cytoscape', 'elements'),
    Input('Next','n_clicks'),
    Input('Previous','n_clicks'))
def update_tree(nextP, previousP):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'Next' or trigger == 'Previous':
        idP = max(0, nextP-previousP)
        result = pre_visu(idP)
        tab = result[0]

        elements = []
        element = other(tab, 0, elements, k=0, u=0, nLeaf=0, selected=-1)
        return element[0]
    else:

        result = pre_visu(0)
        tab = result[0]
        elements = []
        element = other(tab, 0, elements, k=0, u=0, nLeaf=0, selected=-1)
        return element[0]


@app.callback(
    Output('basic-interactions', 'figure'),
    Input('cytoscape', 'mouseoverNodeData'),
    Input('Next', 'n_clicks'),
    Input('Previous', 'n_clicks'))
def update_time_line(data, nextP, previousP):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'cytoscape':
        idP = max(0, nextP-previousP)
        result = pre_visu(idP)
        row = result[3]
        df = result[4]
        idClicked = 0
        compteur = 0
        while compteur <= data['leaf'] and data['leaf']>0:
            if not row[0][idClicked] == '':
                compteur = compteur + 1
            idClicked = idClicked + 1
        if idClicked >0:
            idClicked = idClicked - 1
        if idClicked == 0 and row[0][idClicked] == '':
            idClicked = idClicked + 1
        for rows in range(df.shape[0]):
            if int(df.at[rows, 'id']) == idClicked:
                df.at[rows,'dot'] = 20
            else:
                df.at[rows,'dot'] = 10

        figure = px.scatter(df, x="x", y="y", color="id", color_discrete_sequence=px.colors.qualitative.Alphabet, hover_name="label", size="dot",
                         hover_data={
                             "y": False,
                             "id":False,
                             "ideal":True,
                             "nRep": True,
                             "dot":False
                         })
        pover = 1
        minMax = [df.at[0, "x"]]
        if not len(row[0]) == 1:
            for r in range(1, df.shape[0]):
                if int(df.at[r, "nRep"]) == int(df.at[r-1, "nRep"]):
                    minMax.append(df.at[r,"x"])
                else:
                    figure.add_shape(type='line',
                                         x0=min(minMax),
                                         y0=0.07 - pover * 0.05,
                                         x1=max(minMax),
                                         y1=0.07 - pover * 0.05,
                                         line=dict(color="limegreen", ),
                                         xref='x',
                                         yref='y'
                                         )
                    #Can be uncomment to have vertical lines at occurrences end
                    #but it increase the computation time
                    """figure.add_shape(type='line',
                                     x0=max(minMax),
                                     y0=1 - pover * 0.05,
                                     x1=max(minMax),
                                     y1=-1 - pover * 0.05,
                                     line=dict(color="salmon", dash="dash"),
                                     xref='x',
                                     yref='y'
                                     )"""
                    pover = pover + 1
                    minMax = [df.at[r,"x"]]
            figure.add_shape(type='line',
                      x0=min(minMax),
                      y0=0.07 - pover*0.05,
                      x1=max(minMax),
                      y1=0.07 - pover*0.05,
                      line=dict(color="limegreen", ),
                      xref='x',
                      yref='y'
                      )
            """figure.add_shape(type='line',
                      x0=max(minMax),
                      y0=1 - pover*0.05,
                      x1=max(minMax),
                      y1=-1 - pover*0.05,
                      line=dict(color="salmon",dash="dash"),
                      xref='x',
                      yref='y'
                      )"""
        figure.update_layout(clickmode='event+select')
        figure.update_layout(showlegend=False)
        return figure

    elif trigger == "Next" or trigger == "Previous":
        idP = max(0, nextP-previousP)
        result = pre_visu(idP)
        row = result[3]
        df = result[4]
        figure = px.scatter(df, x="x", y="y", color="id", color_discrete_sequence=px.colors.qualitative.Alphabet, hover_name="label", size="size",
                         hover_data={
                             "y": False,
                             "id":False,
                             "ideal":True,
                             "nRep": True,
                         })
        pover = 1
        minMax = [df.at[0, "x"]]
        if not len(row[0]) == 1:
            for r in range(1, df.shape[0]):
                if int(df.at[r, "nRep"]) == int(df.at[r-1, "nRep"]):
                    minMax.append(df.at[r,"x"])
                else:
                    figure.add_shape(type='line',
                                         x0=min(minMax),
                                         y0=0.07 - pover * 0.05,
                                         x1=max(minMax),
                                         y1=0.07 - pover * 0.05,
                                         line=dict(color="limegreen", ),
                                         xref='x',
                                         yref='y'
                                         )
                    #Can be uncomment to have vertical lines at occurrences end
                    #but it increase the computation time
                    """figure.add_shape(type='line',
                                     x0=max(minMax),
                                     y0=1 - pover * 0.05,
                                     x1=max(minMax),
                                     y1=-1 - pover * 0.05,
                                     line=dict(color="salmon", dash="dash"),
                                     xref='x',
                                     yref='y'
                                     )"""
                    pover = pover + 1
                    minMax = [df.at[r,"x"]]
            figure.add_shape(type='line',
                      x0=min(minMax),
                      y0=0.07 - pover*0.05,
                      x1=max(minMax),
                      y1=0.07 - pover*0.05,
                      line=dict(color="limegreen", ),
                      xref='x',
                      yref='y'
                      )
            """figure.add_shape(type='line',
                      x0=max(minMax),
                      y0=1 - pover*0.05,
                      x1=max(minMax),
                      y1=-1 - pover*0.05,
                      line=dict(color="salmon",dash="dash"),
                      xref='x',
                      yref='y'
                      )"""
        figure.update_layout(clickmode='event+select')
        figure.update_layout(showlegend=False)
        return figure

    else:

        result = pre_visu(0)
        df = result[4]
        row = result[3]
        figure = px.scatter(df, x="x", y="y", color="id", color_discrete_sequence=px.colors.qualitative.Alphabet, hover_name="label",
                         hover_data={
                             "y": False,
                             "id":False,
                             "ideal":True
                         },)

        pover = 1

        if not len(row[0]) == 1:
            for r in range(1, df.shape[0]):
                if int(df.at[r, "nRep"]) == int(df.at[r-1, "nRep"]):
                    minMax.append(df.at[r,"x"])
                else:
                    figure.add_shape(type='line',
                                         x0=min(minMax),
                                         y0=0.07 - pover * 0.05,
                                         x1=max(minMax),
                                         y1=0.07 - pover * 0.05,
                                         line=dict(color="limegreen", ),
                                         xref='x',
                                         yref='y'
                                         )
                    #Can be uncomment to have vertical lines at occurrences end
                    #but it increase the computation time
                    """figure.add_shape(type='line',
                                     x0=max(minMax),
                                     y0=1 - pover * 0.05,
                                     x1=max(minMax),
                                     y1=-1 - pover * 0.05,
                                     line=dict(color="salmon", dash="dash"),
                                     xref='x',
                                     yref='y'
                                     )"""
                    pover = pover + 1
                    minMax = [df.at[r,"x"]]
            print("lines")
            figure.add_shape(type='line',
                      x0=min(minMax),
                      y0=0.07 - pover*0.05,
                      x1=max(minMax),
                      y1=0.07 - pover*0.05,
                      line=dict(color="limegreen", ),
                      xref='x',
                      yref='y'
                      )
            """figure.add_shape(type='line',
                      x0=max(minMax),
                      y0=1 - pover*0.05,
                      x1=max(minMax),
                      y1=-1 - pover*0.05,
                      line=dict(color="salmon",dash="dash"),
                      xref='x',
                      yref='y'
                      )"""
        figure.update_layout(clickmode='event+select')
        figure.update_layout(showlegend=False)
        return figure

@app.callback(
    Output('t0', 'children'),
    Input('Next', 'n_clicks'),
    Input('Previous', 'n_clicks'))
def button_trigger(nextP, previousP):
    idP = max(0, nextP - previousP)
    result = pre_visu(idP)
    t0 = result[1]
    return t0

@app.callback(
    Output('pattern id', 'children'),
    Input('Next', 'n_clicks'),
    Input('Previous', 'n_clicks'))
def button_trigger(nextP, previousP):
    idP = max(0, nextP - previousP)
    numberP = "Patterns id : " + str(idP)
    return numberP


if __name__ == "__main__":
    f = sys.argv[1]
    file = open(f)
    line = file.readlines()
    while not line[0][0] == "t":
        line.pop(0)
    patternID = 0
    result = pre_visu(0)
    tab = result[0]
    t0 = result[1]
    dE = result[2]
    row = result[3]
    df = result[4]
    i = result[5]
    minus = []
    pos = []

    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    app.layout = html.Div(
        [html.H1( id="t0"),
         cyto.Cytoscape(
             id='cytoscape',
             style={'width': '100%', 'height': '600px'},
             layout={'name': 'preset'},
             stylesheet=[
                 {'selector': 'node', 'style': {'label': 'data(label)',
                                                'text-valign': 'top'}},
                 {'selector': 'edge', 'style': {
                     'curve-style': 'bezier'
                 }
                  },
                 {'selector': '.labelled',
                  'style': {
                      'label': 'data(label)',
                      'line-style': 'dotted',
                      'curve-style': 'bezier',
                      'target-arrow-shape': 'triangle',
                      'arrow-scale': 1
                  },
                  }
             ]
         ),
         html.Div([
             dcc.Graph(
                 id='basic-interactions'
             ), ]),
         html.Div([
             html.Button('Previous', id='Previous', n_clicks=0),
             html.Button('Next', id='Next', n_clicks=0),
             html.H2(id="pattern id"), ])
         ]
    )
    app.run_server(debug=False)
