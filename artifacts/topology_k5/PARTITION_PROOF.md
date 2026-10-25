# Partition proof

Every internal node below lists the cut that was chosen and the staging path that was built.
Connectedness was checked on the induced subgraph, not assumed from the drawing.

## Node d0_0

size: 105
bounding box: rows [0, 12], columns [0, 14]
connected: yes
split: vertical at 7
left size: 59
right size: 46
left connected: yes
right connected: yes
cut edges: 13
staging path: (3,3) -> (3,4) -> (3,5) -> (3,6) -> (3,7) -> (3,8) -> (4,8) -> (5,8) -> (6,8) -> (7,8)
staging path valid: yes
left child bounding box: rows [0, 12], columns [0, 7]
right child bounding box: rows [0, 12], columns [8, 14]

## Node d1_0

size: 59
bounding box: rows [0, 12], columns [0, 7]
connected: yes
split: horizontal at 5
left size: 27
right size: 32
left connected: yes
right connected: yes
cut edges: 7
staging path: (1,5) -> (2,5) -> (3,5) -> (4,5) -> (5,5) -> (6,5) -> (6,4) -> (6,3) -> (6,2) -> (6,1)
staging path valid: yes
left child bounding box: rows [0, 5], columns [1, 7]
right child bounding box: rows [6, 12], columns [0, 7]

## Node d2_2

size: 27
bounding box: rows [0, 5], columns [1, 7]
connected: yes
split: horizontal at 3
left size: 14
right size: 13
left connected: yes
right connected: yes
cut edges: 5
staging path: (3,7) -> (3,6) -> (3,5) -> (3,4) -> (3,3) -> (4,3) -> (4,4) -> (4,5) -> (4,6) -> (4,7)
staging path valid: yes
left child bounding box: rows [0, 3], columns [3, 7]
right child bounding box: rows [4, 5], columns [1, 7]

## Node d3_4

size: 14
bounding box: rows [0, 3], columns [3, 7]
connected: yes
split: vertical at 5
left size: 6
right size: 8
left connected: yes
right connected: yes
cut edges: 3
staging path: (3,3) -> (3,4) -> (2,4) -> (2,5) -> (3,5) -> (3,6) -> (3,7) -> (2,7) -> (2,6) -> (1,6)
staging path valid: yes
left child bounding box: rows [1, 3], columns [3, 5]
right child bounding box: rows [0, 3], columns [6, 7]

## Node d4_6

size: 6
bounding box: rows [1, 3], columns [3, 5]
connected: yes
kind: leaf
leaf anchor: (3,3) -> (3,4) -> (2,4) -> (2,5) -> (3,5)

## Node d4_7

size: 8
bounding box: rows [0, 3], columns [6, 7]
connected: yes
kind: leaf
leaf anchor: (3,6) -> (3,7) -> (2,7) -> (2,6) -> (1,6)

## Node d3_5

size: 13
bounding box: rows [4, 5], columns [1, 7]
connected: yes
split: vertical at 4
left size: 7
right size: 6
left connected: yes
right connected: yes
cut edges: 2
staging path: (5,2) -> (4,2) -> (4,3) -> (4,4) -> (5,4) -> (5,5) -> (4,5) -> (4,6) -> (4,7) -> (5,7)
staging path valid: yes
left child bounding box: rows [4, 5], columns [1, 4]
right child bounding box: rows [4, 5], columns [5, 7]

## Node d4_8

size: 7
bounding box: rows [4, 5], columns [1, 4]
connected: yes
kind: leaf
leaf anchor: (5,2) -> (4,2) -> (4,3) -> (4,4) -> (5,4)

## Node d4_9

size: 6
bounding box: rows [4, 5], columns [5, 7]
connected: yes
kind: leaf
leaf anchor: (5,5) -> (4,5) -> (4,6) -> (4,7) -> (5,7)

## Node d2_3

size: 32
bounding box: rows [6, 12], columns [0, 7]
connected: yes
split: horizontal at 7
left size: 14
right size: 18
left connected: yes
right connected: yes
cut edges: 6
staging path: (7,6) -> (7,5) -> (7,4) -> (7,3) -> (7,2) -> (8,2) -> (8,3) -> (8,4) -> (8,5) -> (8,6)
staging path valid: yes
left child bounding box: rows [6, 7], columns [0, 7]
right child bounding box: rows [8, 12], columns [2, 7]

## Node d3_10

size: 14
bounding box: rows [6, 7], columns [0, 7]
connected: yes
split: vertical at 3
left size: 6
right size: 8
left connected: yes
right connected: yes
cut edges: 2
staging path: (6,0) -> (6,1) -> (6,2) -> (6,3) -> (7,3) -> (7,4) -> (6,4) -> (6,5) -> (7,5) -> (7,6)
staging path valid: yes
left child bounding box: rows [6, 7], columns [0, 3]
right child bounding box: rows [6, 7], columns [4, 7]

## Node d4_12

size: 6
bounding box: rows [6, 7], columns [0, 3]
connected: yes
kind: leaf
leaf anchor: (6,0) -> (6,1) -> (6,2) -> (6,3) -> (7,3)

## Node d4_13

size: 8
bounding box: rows [6, 7], columns [4, 7]
connected: yes
kind: leaf
leaf anchor: (7,4) -> (6,4) -> (6,5) -> (7,5) -> (7,6)

## Node d3_11

size: 18
bounding box: rows [8, 12], columns [2, 7]
connected: yes
split: vertical at 5
left size: 8
right size: 10
left connected: yes
right connected: yes
cut edges: 3
staging path: (8,2) -> (8,3) -> (8,4) -> (8,5) -> (9,5) -> (9,6) -> (9,7) -> (10,7) -> (11,7) -> (12,7)
staging path valid: yes
left child bounding box: rows [8, 10], columns [2, 5]
right child bounding box: rows [8, 12], columns [6, 7]

## Node d4_14

size: 8
bounding box: rows [8, 10], columns [2, 5]
connected: yes
kind: leaf
leaf anchor: (8,2) -> (8,3) -> (8,4) -> (8,5) -> (9,5)

## Node d4_15

size: 10
bounding box: rows [8, 12], columns [6, 7]
connected: yes
kind: leaf
leaf anchor: (9,6) -> (9,7) -> (10,7) -> (11,7) -> (12,7)

## Node d1_1

size: 46
bounding box: rows [0, 12], columns [8, 14]
connected: yes
split: vertical at 9
left size: 23
right size: 23
left connected: yes
right connected: yes
cut edges: 9
staging path: (6,9) -> (5,9) -> (4,9) -> (3,9) -> (2,9) -> (2,10) -> (3,10) -> (4,10) -> (5,10) -> (6,10)
staging path valid: yes
left child bounding box: rows [0, 12], columns [8, 9]
right child bounding box: rows [2, 10], columns [10, 14]

## Node d2_16

size: 23
bounding box: rows [0, 12], columns [8, 9]
connected: yes
split: horizontal at 6
left size: 12
right size: 11
left connected: yes
right connected: yes
cut edges: 2
staging path: (2,9) -> (3,9) -> (4,9) -> (5,9) -> (6,9) -> (7,9) -> (8,9) -> (9,9) -> (10,9) -> (11,9)
staging path valid: yes
left child bounding box: rows [0, 6], columns [8, 9]
right child bounding box: rows [7, 12], columns [8, 9]

## Node d3_18

size: 12
bounding box: rows [0, 6], columns [8, 9]
connected: yes
split: horizontal at 3
left size: 6
right size: 6
left connected: yes
right connected: yes
cut edges: 2
staging path: (0,8) -> (1,8) -> (2,8) -> (2,9) -> (3,9) -> (4,9) -> (4,8) -> (5,8) -> (5,9) -> (6,9)
staging path valid: yes
left child bounding box: rows [0, 3], columns [8, 9]
right child bounding box: rows [4, 6], columns [8, 9]

## Node d4_20

size: 6
bounding box: rows [0, 3], columns [8, 9]
connected: yes
kind: leaf
leaf anchor: (0,8) -> (1,8) -> (2,8) -> (2,9) -> (3,9)

## Node d4_21

size: 6
bounding box: rows [4, 6], columns [8, 9]
connected: yes
kind: leaf
leaf anchor: (4,9) -> (4,8) -> (5,8) -> (5,9) -> (6,9)

## Node d3_19

size: 11
bounding box: rows [7, 12], columns [8, 9]
connected: yes
split: horizontal at 9
left size: 6
right size: 5
left connected: yes
right connected: yes
cut edges: 2
staging path: (7,8) -> (7,9) -> (8,9) -> (9,9) -> (9,8) -> (10,8) -> (10,9) -> (11,9) -> (11,8) -> (12,8)
staging path valid: yes
left child bounding box: rows [7, 9], columns [8, 9]
right child bounding box: rows [10, 12], columns [8, 9]

## Node d4_22

size: 6
bounding box: rows [7, 9], columns [8, 9]
connected: yes
kind: leaf
leaf anchor: (7,8) -> (7,9) -> (8,9) -> (9,9) -> (9,8)

## Node d4_23

size: 5
bounding box: rows [10, 12], columns [8, 9]
connected: yes
kind: leaf
leaf anchor: (10,8) -> (10,9) -> (11,9) -> (11,8) -> (12,8)

## Node d2_17

size: 23
bounding box: rows [2, 10], columns [10, 14]
connected: yes
split: horizontal at 6
left size: 13
right size: 10
left connected: yes
right connected: yes
cut edges: 4
staging path: (2,10) -> (3,10) -> (4,10) -> (5,10) -> (6,10) -> (7,10) -> (7,11) -> (7,12) -> (8,12) -> (8,11)
staging path valid: yes
left child bounding box: rows [2, 6], columns [10, 14]
right child bounding box: rows [7, 10], columns [10, 13]

## Node d3_24

size: 13
bounding box: rows [2, 6], columns [10, 14]
connected: yes
split: vertical at 10
left size: 5
right size: 8
left connected: yes
right connected: yes
cut edges: 3
staging path: (2,10) -> (3,10) -> (4,10) -> (5,10) -> (6,10) -> (6,11) -> (5,11) -> (4,11) -> (4,12) -> (5,12)
staging path valid: yes
left child bounding box: rows [2, 6], columns [10, 10]
right child bounding box: rows [4, 6], columns [11, 14]

## Node d4_26

size: 5
bounding box: rows [2, 6], columns [10, 10]
connected: yes
kind: leaf
leaf anchor: (2,10) -> (3,10) -> (4,10) -> (5,10) -> (6,10)

## Node d4_27

size: 8
bounding box: rows [4, 6], columns [11, 14]
connected: yes
kind: leaf
leaf anchor: (6,11) -> (5,11) -> (4,11) -> (4,12) -> (5,12)

## Node d3_25

size: 10
bounding box: rows [7, 10], columns [10, 13]
connected: yes
kind: leaf
leaf anchor: (7,10) -> (7,11) -> (7,12) -> (8,12) -> (8,11)
