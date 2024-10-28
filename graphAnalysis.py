# Import Libraries
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc

# Create SparkSession with GraphFrames
# สร้าง SparkSession พร้อม GraphFrames (กำหนด memory ของ driver เป็น 4GB และเพิ่ม package graphframes ที่ใช้ version ที่เข้ากันได้กับ Spark 3.0)
spark = SparkSession.builder \
    .appName("GraphAnalytics") \
    .config("spark.driver.memory", "4g") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# Create vertices DataFrame
# สร้าง DataFrame ของ vertices (จุดในกราฟ) พร้อมกับระบุข้อมูล 'id' (ชื่อ) และ 'age' (อายุ)
vertices = spark.createDataFrame([
    ("Alice", "45"),
    ("Jacob", "43"),
    ("Roy", "21"),
    ("Ryan", "49"),
    ("Emily", "24"),
    ("Sheldon", "52"),
], ["id", "age"])

# Create edges DataFrame
# สร้าง DataFrame ของ edges (เส้นเชื่อมในกราฟ) โดยระบุ 'src' (จุดเริ่ม), 'dst' (จุดปลาย), และ 'relation' (ความสัมพันธ์)
edges = spark.createDataFrame([
    ("Sheldon", "Alice", "Sister"),
    ("Alice", "Jacob", "Husband"),
    ("Emily", "Jacob", "Father"),
    ("Ryan", "Alice", "Friend"),
    ("Alice", "Emily", "Daughter"),
    ("Alice", "Roy", "Son"),
    ("Jacob", "Roy", "Son"),
], ["src", "dst", "relation"])

# Create GraphFrame
# สร้างกราฟด้วย GraphFrame โดยใช้ vertices และ edges ที่สร้างไว้ก่อนหน้า
try:
    graph = GraphFrame(vertices, edges)
    print("GraphFrame created successfully.")
except Exception as e:
    print(f"Error creating GraphFrame: {e}")

# Show vertices and edges
# แสดงข้อมูล vertices และ edges ที่มีอยู่ในกราฟ
print("Vertices:")
vertices.show()

print("Edges:")
edges.show()

# Group and order the nodes and edges
# จัดกลุ่ม edges ตาม 'src' และ 'dst' พร้อมทั้งนับจำนวนและเรียงลำดับตามจำนวนที่สูงสุด
grouped_edges = graph.edges.groupBy("src", "dst").count().orderBy(desc("count"))
print("Grouped Edges:")
grouped_edges.show(5)

# Filter the GraphFrame
# กรอง edges เพื่อแสดงเฉพาะเส้นเชื่อมที่ 'src' เป็น 'Alice' หรือ 'dst' เป็น 'Jacob' และเรียงลำดับตามจำนวนที่มากที่สุด
filtered_edges = graph.edges.where("src = 'Alice' OR dst = 'Jacob'").groupBy("src", "dst").count().orderBy(desc("count"))
print("Filtered Edges:")
filtered_edges.show(5)

# Create a subgraph
# สร้าง subgraph ที่กรองเฉพาะ edges ที่ 'src' เป็น 'Alice' หรือ 'dst' เป็น 'Jacob'
subgraph_query = graph.edges.where("src = 'Alice' OR dst = 'Jacob'")
subgraph = GraphFrame(graph.vertices, subgraph_query)
print("Subgraph Edges:")
subgraph.edges.show()

# Find motifs
# ค้นหา motifs (รูปแบบในกราฟ) โดยใช้ syntax `(a) - [ab] -> (b)` เพื่อค้นหาความสัมพันธ์ระหว่างจุด 'a' และ 'b'
motifs = graph.find("(a) - [ab] -> (b)")
print("Motifs:")
motifs.show()

# PageRank
# ใช้ PageRank algorithm เพื่อคำนวณคะแนนของจุดในกราฟ โดยใช้ค่า resetProbability เป็น 0.15 และวนซ้ำ 5 รอบ
rank = graph.pageRank(resetProbability=0.15, maxIter=5)
print("PageRank:")
rank.vertices.orderBy(desc("pagerank")).show(5)

# In-Degree and Out-Degree
# คำนวณ in-degree (จำนวน edges ที่เข้าสู่แต่ละจุด) และเรียงลำดับตามจำนวนสูงสุด
in_degree = graph.inDegrees
print("In-Degree:")
in_degree.orderBy(desc("inDegree")).show(5)

# คำนวณ out-degree (จำนวน edges ที่ออกจากแต่ละจุด) และเรียงลำดับตามจำนวนสูงสุด
out_degree = graph.outDegrees
print("Out-Degree:")
out_degree.orderBy(desc("outDegree")).show(5)

# Connected Components
# ค้นหา connected components (กลุ่มของจุดที่เชื่อมต่อถึงกัน) ในกราฟ
try:
    connected_components = graph.connectedComponents()
    print("Connected Components:")
    connected_components.show()
except Exception as e:
    print(f"Error calculating connected components: {e}")

# Strongly Connected Components
# ค้นหา strongly connected components (กลุ่มของจุดที่มีการเชื่อมต่อถึงกันทั้งไปและกลับ) โดยใช้การวนซ้ำ 5 รอบ
scc = graph.stronglyConnectedComponents(maxIter=5)
print("Strongly Connected Components:")
scc.show()

# Breadth-First Search (BFS)
# ค้นหาเส้นทางที่สั้นที่สุดจากจุดที่ id เป็น 'Alice' ไปยังจุดที่ id เป็น 'Roy' โดยกำหนดความยาวสูงสุดของเส้นทางเป็น 2
bfs_result = graph.bfs(fromExpr="id = 'Alice'", toExpr="id = 'Roy'", maxPathLength=2)
print("BFS Result from id 'Alice' to id 'Roy' with maxPathLength = 2:")
bfs_result.show()

# Stop SparkSession when done
# หยุดการทำงานของ SparkSession หลังจากเสร็จสิ้นกระบวนการทั้งหมด
spark.stop()
