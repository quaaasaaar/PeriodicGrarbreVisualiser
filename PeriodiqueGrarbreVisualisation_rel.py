import sys

import dash
import dash_cytoscape as cyto
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Output, Input, State
import re


def other(tab,i,elements,k=0,u=0, nLeaf=0, selected=-1):
    tmp = []
    rx = re.search('\((.*)\)\[(.*)]',tab)
    r = rx[1].split(sep=' ')
    elements = elements + [{
                'data': {'id': str(i), 'label': rx[2]},
                'position': {'x': 100 * i, 'y': 50 * k},
            },]
    j = i
    i = i + 1
    k = i
    rm = 0
    for elem in range(0, len(r)):
        if rm > 0:
            rm = rm + len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp[len(tmp)-1] = tmp[len(tmp)-1] + ' ' + r[elem]
        elif r[elem][0] == 'p':
            tmp[len(tmp)-1] = tmp[len(tmp)-1] + ' ' + r[elem]
        elif r[elem][0] == '(':
            rm = len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp.append(r[elem])


        else:
            tmp.append(r[elem])

    for e in range(0, len(tmp)):
        if not tmp[e][0] == '(' and not tmp[e][0] == '[':
            y = i -1
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
            nLeaf = nLeaf+1
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

        if e < len(tmp)-1 and tmp[e+1][1] == 'd':
            if tmp[e][0] == '(':
                elements = elements + [{
                    'data': {'source': str(u), 'target': str(i), 'label': tmp[e + 1]},
                    'classes': 'labelled'
                }]
            else:
                elements = elements + [{
                        'data': {'source': str(i-1), 'target': str(i), 'label': tmp[e+1]},
                        'classes': 'labelled'
                    }]
    t = [elements]
    t.append(i)
    t.append(nLeaf)
    return t

# We create a dataFrame with id, label, period, rep, nested levels, deltas we don't care about shifts yet
def createDataFrame(tab, i, row, k=1, u=0):
    tmp = []
    rx = re.search('\((.*)\)\[r=(\d*) p=(\d*)]',tab)
    r = rx[1].split(sep=' ')
    ## Create a row with p and r and nN = 1 and d = 0
    row[0].append('')
    row[1].append(int(rx[3]))
    row[2].append(int(rx[2]))
    row[3].append(k)
    row[4].append(0)
    #Do not change this for : we will use this format after for our things
    rm = 0
    for elem in range(0, len(r)):
        if rm > 0:
            rm = rm + len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp[len(tmp)-1] = tmp[len(tmp)-1] + ' ' + r[elem]
        elif r[elem][0] == 'p':
            tmp[len(tmp)-1] = tmp[len(tmp)-1] + ' ' + r[elem]
        elif r[elem][0] == '(':
            rm = len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp.append(r[elem])


        else:
            tmp.append(r[elem])
    #In this for :
                #   add the label at the right place (last row)
                #   add a row for each d =
    for e in range(0, len(tmp)):
        if not tmp[e][0] == '(' and not tmp[e][0] == '[':
            row[0][len(row[0])-1] = tmp[e]

        if tmp[e][0] == '(':
            k = k+1
            row = createDataFrame(tmp[e],i,row,k,u)
            k = k-1

        if tmp[e][1] == 'd':
            te = re.search('d=(\d*)',tmp[e])
            row[0].append('')
            row[1].append(0)
            row[2].append(0)
            row[3].append(k)
            row[4].append(int(te[1]))

    return row

def computeShiftsNested(row, tabE, tabt):
    t0 = re.search('=(\d+)', tabt)
    t0 = int(t0[1])
    lE = re.split(' ', tabE)
    #Tabs to create DataFrame we insert first elem later
    x = []
    y = []
    label = []
    ids = []
    size = []
    for elem in range(len(row[2])):
        if row[3][elem] == 1:
            for nRep in range(row[2][elem]):
                    i = elem+1
                    tmpTime = t0 + row[1][elem] * nRep + int(lE[len(x) - 1])
                    if not row[0][elem] == '':
                        x.append(tmpTime)
                        y.append(1 - row[3][elem] - nRep*0.05)
                        ids.append(str(elem))
                        label.append(row[0][elem])
                        size.append(12)

                    while i < len(row[2]):
                        if row[3][i] > row[3][elem]:
                            nLevels(row, t0, lE, x, y, label, ids, i, tmpTime, row[4][i-1])
                        elif not row[0][i] == '':
                            x.append(x[len(x) - 1] + row[4][i] + int(lE[len(x)-1]))
                            y.append(1 - row[3][i] - nRep*0.05)
                            ids.append(str(i))
                            label.append(row[0][i])
                            size.append(12)

                        i = i + 1

    while len(size) < len(x):
        size.append(12)
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "id": ids,
        "label": label,
        "size": size
    })
    return df

def nLevels(row, t0, lE, x, y, label, ids, elem, tmpTime, dPred=0):
    for nRep in range(row[2][elem]):
        i = elem + 1
        #Reminder : tmpTime = t0 + row[1][elem] * nRep + int(lE[len(x) - 1])
        if not row[0][elem] == '':
            x.append(tmpTime + row[1][elem] * nRep + int(lE[len(x) - 1]) + dPred)
            y.append(1 - row[3][elem])
            ids.append(str(elem))
            label.append(row[0][elem])

        while i < len(row[2]):
            if row[3][i] > row[3][elem]:
                nLevels(row, t0, lE, x, y, label, ids, i, tmpTime, row[4][i-1])
            i = i + 1

def splitshift(tab,pos,minus):
    dE = re.split(' ', tab)
    dE.insert(0,0)
    for e in dE:
        if int(e) == 0:
            pos.append(e)
            minus.append(e)
        elif int(e) < 0:
            pos.append(0)
            minus.append(-int(e))
        elif int(e) > 0:
            pos.append(e)
            minus.append(0)

app = dash.Dash(__name__)




@app.callback(
    Output('basic-interactions', 'figure'),
    Input('cytoscape', 'mouseoverNodeData'))
def update_time_line(data):
    row = createDataFrame(tab[1], i, [[],[],[],[],[]])
    df = computeShiftsNested(row, tab[6], tab[0])
    if data:
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

        figure = px.scatter(df, x="x", y="y", color="id", color_discrete_sequence=px.colors.qualitative.Alphabet, hover_name="label", error_x=minus, error_x_minus=pos, size="dot",
                         hover_data={
                             "y": False,
                             "id":False
                         })
        pover = 1
        xMin = int(df.at[0, "x"])
        xMax = 0
        temp = int(df.at[0, "x"])
        for r in range(1, df.shape[0]):
            if int(df.at[r, "id"]) == 0:
                xMin = temp
                temp = int(df.at[r, "x"])
            if int(df.at[r, "id"]) < int(df.at[r - 1, "id"]) or int(df.at[r, "id"]) == 0:
                xMax = int(df.at[r - 1, "x"])
                figure.add_vline(x=xMax, line_width=3, line_dash="dash", line_color="salmon",
                                 annotation_text=pover)
                figure.add_shape(type='line',
                                     x0=xMin,
                                     y0=0.07 - pover * 0.05,
                                     x1=xMax,
                                     y1=0.07 - pover * 0.05,
                                     line=dict(color="limegreen", ),
                                     xref='x',
                                     yref='y'
                                     )
                pover = pover + 1
                xMin = int(df.at[r,"x"])

        figure.add_shape(type='line',
                      x0=xMin,
                      y0=0.07 - pover*0.05,
                      x1=df.at[r,"x"],
                      y1=0.07 - pover*0.05,
                      line=dict(color="limegreen", ),
                      xref='x',
                      yref='y'
                      )
        figure.add_vline(x=df.at[df.shape[0]-1, "x"], line_width=3, line_dash="dash", line_color="salmon", annotation_text=pover)
        figure.update_layout(clickmode='event+select')
        figure.update_layout(showlegend=False)
        return figure

    else:
        figure = px.scatter(df, x="x", y="y", color="id", color_discrete_sequence=px.colors.qualitative.Alphabet, hover_name="label", error_x=minus, error_x_minus=pos,
                         hover_data={
                             "y": False,
                             "id":False,
                         },)

        pover = 1
        xMin = int(df.at[0,"x"])
        xMax = 0
        temp = int(df.at[0,"x"])
        for r in range(1,df.shape[0]):
            if int(df.at[r,"id"]) == 0:
                xMin = temp
                temp = int(df.at[r,"x"])
            if int(df.at[r,"id"]) < int(df.at[r-1,"id"]) or int(df.at[r,"id"]) == 0:
                xMax = int(df.at[r-1,"x"])
                figure.add_vline(x=xMax, line_width=3, line_dash="dash", line_color="salmon", annotation_text=pover)
                figure.add_shape(type='line',
                                     x0=xMin,
                                     y0=0.07 - pover*0.05,
                                     x1=xMax,
                                     y1=0.07 - pover*0.05,
                                     line=dict(color="limegreen", ),
                                     xref='x',
                                     yref='y'
                                     )
                xMin = int(df.at[r, "x"])
                pover = pover + 1
        figure.add_shape(type='line',
                      x0=xMin,
                      y0=0.07 - pover*0.05,
                      x1=df.at[r,"x"],
                      y1=0.07 - pover*0.05,
                      line=dict(color="limegreen", ),
                      xref='x',
                      yref='y'
                      )
        figure.add_vline(x=df.at[df.shape[0]-1, "x"], line_width=3, line_dash="dash", line_color="salmon", annotation_text=pover)
        figure.update_layout(clickmode='event+select')
        figure.update_layout(showlegend=False)
        return figure

if __name__ == "__main__":
    # Concat
    #t = "t0=755	(Discretionary-Unpaidwork-Subway [d=1053] Discretionary-Unpaidwork-Cook [d=1941] Discretionary-Unpaidwork-Cook [d=4902] Discretionary-Productive-Writing)[r=4 p=6915]	Code length:158.552276	sum(|E|)=14\tOccs (16/16)	concat\t0 0 0 0 -1 1 0 0 -1 0 -1 2 3 -3 -2"

    # Simple
    # t = "t0=15663	(Sleep)[r=45 p=2]	Code length:213.489824	sum(|E|)=82	Occs (45/45)	simple	1 -1 2 3 1 0 0 4 -1 3 0 10 5 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 4 7 3 7 2 1 2 2 0 8 0 7 0 5 0	X"

    # Nested
    #t = "t0=33140	((Business-Consulting-E1_General)[r=4 p=1])[r=3 p=155]	Code length:97.088756	sum(|E|)=1	Occs (12/12)	nested	0 0 0 0 0 0 0 1 0 0 0	X"

    # Other
    # t = "t0=6421	(Business-Connect-Connecting [d=193] ((Business-Consulting-E1_General [d=69] Discretionary-Unpaidwork-Laundry)[r=4 p=1])[r=3 p=155] [d=1832] Discretionary-Productive-Drawing)[r=3 p=412]	Code length:147.571622	sum(|E|)=26 Occs (15/15)	other\t0 0 0 0 -1 1 0 0 -1 0 -1 2 3 -3 -2 0 0 0 0 -1 1 0 0 -1 0 -1 2 3 -3 -2 0 0 0 0 -1 1 0 0 -1 0 -1 2 3 -3 -2	"

    # Other shift
    #t ="t0=7091	((Discretionary-Productive-Coding)[r=5 p=2] [d=445] Discretionary-Productive-Emacs)[r=3 p=14207]	Code length:149.027045	sum(|E|)=29	Occs (18/18)	other	0 0 0 0 1 0 -2 0 1 2 0 1 7 0 0 1 14	X"
    t = sys.argv[1]
    tab = t.split(sep="\t")
    rx = ''
    r = ''
    elements = []

    # For TimeLine Construction
    t0 = re.search('=(\d+)', tab[0])
    x = [int(t0[1])]
    p = re.split(' ', tab[1])
    p0 = re.search('p=(\d+)', p[len(p) - 1])
    dE = re.split(' ', tab[6])
    y = [0]
    df = pd.DataFrame(columns=['label', 'period', 'nbRep', 'nivNes', 'd'])
    row = [[], [], [], [], []]
    custom = [0]
    label = []
    periode = p0[1]
    periode = int(periode)

    i = 0
    elements = []
    row = createDataFrame(tab[1], i, [[], [], [], [], []])
    df = computeShiftsNested(row, tab[6], tab[0])
    pos = []
    minus = []
    splitshift(tab[6], pos, minus)

    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }
    app.layout = html.Div(
        [html.H1("t0 = " + str(t0[1])),
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
        ),

        ]
        )
        ]
    )
    app.run_server(debug=False)
