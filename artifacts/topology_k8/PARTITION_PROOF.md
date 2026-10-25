# Partition proof

Every internal node below lists the cut that was chosen and the staging path that was built.
Connectedness was checked on the induced subgraph, not assumed from the drawing.

## Node d0_0

size: 105
bounding box: rows [0, 12], columns [0, 14]
connected: yes
split: vertical at 6
left size: 46
right size: 59
left connected: yes
right connected: yes
cut edges: 13
staging path: (7,6) -> (6,6) -> (5,6) -> (4,6) -> (3,6) -> (2,6) -> (1,6) -> (0,6) -> (0,7) -> (1,7) -> (2,7) -> (3,7) -> (4,7) -> (5,7) -> (6,7) -> (7,7)
staging path valid: yes
left child bounding box: rows [0, 12], columns [0, 6]
right child bounding box: rows [0, 12], columns [7, 14]

## Node d1_0

size: 46
bounding box: rows [0, 12], columns [0, 6]
connected: yes
split: vertical at 4
left size: 23
right size: 23
left connected: yes
right connected: yes
cut edges: 9
staging path: (9,4) -> (8,4) -> (7,4) -> (6,4) -> (5,4) -> (4,4) -> (3,4) -> (2,4) -> (2,5) -> (3,5) -> (4,5) -> (5,5) -> (6,5) -> (7,5) -> (8,5) -> (9,5)
staging path valid: yes
left child bounding box: rows [2, 10], columns [0, 4]
right child bounding box: rows [0, 12], columns [5, 6]

## Node d2_2

size: 23
bounding box: rows [2, 10], columns [0, 4]
connected: yes
split: horizontal at 5
left size: 13
right size: 10
left connected: yes
right connected: yes
cut edges: 4
staging path: (10,4) -> (9,4) -> (8,4) -> (7,4) -> (6,4) -> (6,3) -> (6,2) -> (6,1) -> (5,1) -> (5,2) -> (4,2) -> (4,3) -> (3,3) -> (3,4) -> (4,4) -> (5,4)
staging path valid: yes
left child bounding box: rows [6, 10], columns [0, 4]
right child bounding box: rows [2, 5], columns [1, 4]

## Node d3_4

size: 13
bounding box: rows [6, 10], columns [0, 4]
connected: yes
kind: leaf
leaf anchor: (10,4) -> (9,4) -> (8,4) -> (7,4) -> (6,4) -> (6,3) -> (6,2) -> (6,1)

## Node d3_5

size: 10
bounding box: rows [2, 5], columns [1, 4]
connected: yes
kind: leaf
leaf anchor: (5,1) -> (5,2) -> (4,2) -> (4,3) -> (3,3) -> (3,4) -> (4,4) -> (5,4)

## Node d2_3

size: 23
bounding box: rows [0, 12], columns [5, 6]
connected: yes
split: horizontal at 5
left size: 11
right size: 12
left connected: yes
right connected: yes
cut edges: 2
staging path: (1,5) -> (2,5) -> (3,5) -> (3,6) -> (4,6) -> (4,5) -> (5,5) -> (5,6) -> (6,6) -> (6,5) -> (7,5) -> (7,6) -> (8,6) -> (8,5) -> (9,5) -> (9,6)
staging path valid: yes
left child bounding box: rows [0, 5], columns [5, 6]
right child bounding box: rows [6, 12], columns [5, 6]

## Node d3_6

size: 11
bounding box: rows [0, 5], columns [5, 6]
connected: yes
kind: leaf
leaf anchor: (1,5) -> (2,5) -> (3,5) -> (3,6) -> (4,6) -> (4,5) -> (5,5) -> (5,6)

## Node d3_7

size: 12
bounding box: rows [6, 12], columns [5, 6]
connected: yes
kind: leaf
leaf anchor: (6,6) -> (6,5) -> (7,5) -> (7,6) -> (8,6) -> (8,5) -> (9,5) -> (9,6)

## Node d1_1

size: 59
bounding box: rows [0, 12], columns [7, 14]
connected: yes
split: horizontal at 6
left size: 32
right size: 27
left connected: yes
right connected: yes
cut edges: 7
staging path: (4,7) -> (4,8) -> (4,9) -> (4,10) -> (4,11) -> (4,12) -> (5,12) -> (6,12) -> (7,12) -> (7,11) -> (7,10) -> (7,9) -> (7,8) -> (7,7) -> (8,7) -> (9,7)
staging path valid: yes
left child bounding box: rows [0, 6], columns [7, 14]
right child bounding box: rows [7, 12], columns [7, 13]

## Node d2_8

size: 32
bounding box: rows [0, 6], columns [7, 14]
connected: yes
split: horizontal at 4
left size: 18
right size: 14
left connected: yes
right connected: yes
cut edges: 6
staging path: (4,8) -> (3,8) -> (2,8) -> (1,8) -> (1,7) -> (2,7) -> (3,7) -> (4,7) -> (5,7) -> (6,7) -> (6,8) -> (6,9) -> (6,10) -> (6,11) -> (6,12) -> (6,13)
staging path valid: yes
left child bounding box: rows [0, 4], columns [7, 12]
right child bounding box: rows [5, 6], columns [7, 14]

## Node d3_10

size: 18
bounding box: rows [0, 4], columns [7, 12]
connected: yes
split: vertical at 8
left size: 10
right size: 8
left connected: yes
right connected: yes
cut edges: 3
staging path: (4,7) -> (3,7) -> (2,7) -> (1,7) -> (1,8) -> (2,8) -> (3,8) -> (4,8) -> (4,9) -> (3,9) -> (2,9) -> (2,10) -> (3,10) -> (4,10) -> (4,11) -> (4,12)
staging path valid: yes
left child bounding box: rows [0, 4], columns [7, 8]
right child bounding box: rows [2, 4], columns [9, 12]

## Node d4_12

size: 10
bounding box: rows [0, 4], columns [7, 8]
connected: yes
kind: leaf
leaf anchor: (4,7) -> (3,7) -> (2,7) -> (1,7) -> (1,8) -> (2,8) -> (3,8) -> (4,8)

## Node d4_13

size: 8
bounding box: rows [2, 4], columns [9, 12]
connected: yes
kind: leaf
leaf anchor: (4,9) -> (3,9) -> (2,9) -> (2,10) -> (3,10) -> (4,10) -> (4,11) -> (4,12)

## Node d3_11

size: 14
bounding box: rows [5, 6], columns [7, 14]
connected: yes
kind: leaf
leaf anchor: (5,7) -> (6,7) -> (6,8) -> (6,9) -> (6,10) -> (6,11) -> (6,12) -> (6,13)

## Node d2_9

size: 27
bounding box: rows [7, 12], columns [7, 13]
connected: yes
split: horizontal at 8
left size: 13
right size: 14
left connected: yes
right connected: yes
cut edges: 5
staging path: (7,13) -> (7,12) -> (7,11) -> (7,10) -> (7,9) -> (7,8) -> (7,7) -> (8,7) -> (9,7) -> (9,8) -> (9,9) -> (9,10) -> (10,10) -> (10,9) -> (10,8) -> (10,7)
staging path valid: yes
left child bounding box: rows [7, 8], columns [7, 13]
right child bounding box: rows [9, 12], columns [7, 11]

## Node d3_14

size: 13
bounding box: rows [7, 8], columns [7, 13]
connected: yes
kind: leaf
leaf anchor: (7,13) -> (7,12) -> (7,11) -> (7,10) -> (7,9) -> (7,8) -> (7,7) -> (8,7)

## Node d3_15

size: 14
bounding box: rows [9, 12], columns [7, 11]
connected: yes
kind: leaf
leaf anchor: (9,7) -> (9,8) -> (9,9) -> (9,10) -> (10,10) -> (10,9) -> (10,8) -> (10,7)
