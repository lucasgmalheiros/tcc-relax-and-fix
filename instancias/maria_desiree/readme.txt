To better understand the files with the parameters of each instance, here we present an example of how to read it. Consider the instance file named "AAA01226_0.dat".

The first line indicates, respectively, the quantity of itens and periods:
[6 12]

The second line indicates the number of plants:
[2]

The next 2 (number of plants) lines indicates the capacity of each plant;

Line 1 = Capacity of plant 1

[1541]

After that, in the matrix of (plants*items) rows and 4 columns: at each line we have, respectively, the production time, setup time, setup cost and production cost.

Line 1 = Plant 1 - item 1 

[1.1  52.0  78.1   2.4]

Line 2 = Plant 1 - item 2

[1.7  54.5 819.8   1.5]

	.
	.
	.

Line 7 = Plant 2 - item 1 

[2.4  74.2 870.5   2.0]

Line 8 = Plant 2 - item 2 

[1.8  51.3 240.3   2.2]

Next, the line:

[0.2   0.3   0.3   0.3   0.3   0.3]   [0.3   0.3   0.3   0.3   0.4   0.3]
	      plant 1                             plant 2

Indicates the inventory costs of the items (they are constants along the planning horizon).

The matrix with 12 rows and 12 columns corresponds to the demands of items on the plants.
 
Line 1 = Period 1

[ 64    53   127   175   143    12 ]  [  154    25    66  117   155   103 ]
 	       plant 1                                  plant 2

The last number (0.26) is the transfer cost of an item from plant 1 to plant 2. This costs is the same from plant 2 to plant 1.
