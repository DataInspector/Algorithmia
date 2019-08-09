#!/usr/bin/env bash

set -o pipefail
set -ex

#
# This example script demonstrates how to communicate with the Driverless AI Scoring Service via HTTP.
# The protocol used is JSON-RPC 2.0.
#

# ------------------------------------------------------------
# Name        Type      Range                                 
# ------------------------------------------------------------
# sepal_len   float32   [4.400000095367432, 7.699999809265137]
# sepal_wid   float32   [2.0, 4.199999809265137]              
# petal_len   float32   [1.0, 6.699999809265137]              
# petal_wid   float32   [0.10000000149011612, 2.5]            
# ------------------------------------------------------------

echo "Scoring individual rows..."

curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id": 1,
  "method": "score",
  "params": {
    "row": {
      "sepal_len": "4.5",
      "sepal_wid": "2.700000047683716",
      "petal_len": "1.899999976158142",
      "petal_wid": "0.30000001192092896"
    }
  }
}
EOF
curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id": 2,
  "method": "score",
  "params": {
    "row": {
      "sepal_len": "4.400000095367432",
      "sepal_wid": "2.299999952316284",
      "petal_len": "1.2999999523162842",
      "petal_wid": "0.10000000149011612"
    }
  }
}
EOF
curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id": 3,
  "method": "score",
  "params": {
    "row": {
      "sepal_len": "5.0",
      "sepal_wid": "2.0",
      "petal_len": "1.2000000476837158",
      "petal_wid": "1.0"
    }
  }
}
EOF
curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id": 4,
  "method": "score",
  "params": {
    "row": {
      "sepal_len": "4.599999904632568",
      "sepal_wid": "2.200000047683716",
      "petal_len": "3.299999952316284",
      "petal_wid": "0.20000000298023224"
    }
  }
}
EOF
curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id": 5,
  "method": "score",
  "params": {
    "row": {
      "sepal_len": "5.199999809265137",
      "sepal_wid": "3.0",
      "petal_len": "1.399999976158142",
      "petal_wid": "1.399999976158142"
    }
  }
}
EOF

echo "Scoring multiple rows..."

curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id": 1,
  "method": "score_batch",
  "params": {
    "rows": [
      {
        "sepal_len": "4.5",
        "sepal_wid": "2.700000047683716",
        "petal_len": "1.899999976158142",
        "petal_wid": "0.30000001192092896"
      },
      {
        "sepal_len": "4.400000095367432",
        "sepal_wid": "2.299999952316284",
        "petal_len": "1.2999999523162842",
        "petal_wid": "0.10000000149011612"
      },
      {
        "sepal_len": "5.0",
        "sepal_wid": "2.0",
        "petal_len": "1.2000000476837158",
        "petal_wid": "1.0"
      },
      {
        "sepal_len": "4.599999904632568",
        "sepal_wid": "2.200000047683716",
        "petal_len": "3.299999952316284",
        "petal_wid": "0.20000000298023224"
      },
      {
        "sepal_len": "5.199999809265137",
        "sepal_wid": "3.0",
        "petal_len": "1.399999976158142",
        "petal_wid": "1.399999976158142"
      }
    ]
  }
}
EOF

echo "Get the input columns"
curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id":1,
  "method":"get_column_names",
  "params":{}
}
EOF

echo "Get the transformed columns"
curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id":1,
  "method":"get_transformed_column_names",
  "params":{}
}
EOF

echo "Get the target labels"
curl http://localhost:9090/rpc --header "Content-Type: application/json" --data @- <<EOF
{
  "id":1,
  "method":"get_target_labels",
  "params":{}
}
EOF