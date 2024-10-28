from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

# สร้าง SparkSession และกำหนดการตั้งค่าเพื่อใช้ GraphFrames
spark = SparkSession.builder \
    .appName("Graph Analytics Assignment") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# อ่านข้อมูลเส้นทางการบินจากไฟล์ CSV เป็น DataFrame
airline_routes_df = spark.read.csv("airline_routes.csv", header=True, inferSchema=True)

# แสดงข้อมูลที่อ่านมาเพื่อดูโครงสร้างของ DataFrame
airline_routes_df.show()

# สร้าง DataFrame ของ vertices จากคอลัมน์ 'source_airport' และตั้งชื่อคอลัมน์เป็น 'id'
vertices_df = airline_routes_df.select("source_airport").withColumnRenamed("source_airport", "id").distinct()

# สร้าง DataFrame ของ edges จากคอลัมน์ 'source_airport' และ 'destination_airport'
# เปลี่ยนชื่อคอลัมน์เป็น 'src' และ 'dst' เพื่อใช้กับ GraphFrame
edges_df = airline_routes_df.select("source_airport", "destination_airport") \
    .withColumnRenamed("source_airport", "src") \
    .withColumnRenamed("destination_airport", "dst")

# แสดงข้อมูล vertices และ edges
vertices_df.show()
edges_df.show()

# สร้าง GraphFrame โดยใช้ vertices และ edges ที่สร้างขึ้น
graph = GraphFrame(vertices_df, edges_df)

# แสดงจำนวน vertices และ edges ในกราฟ
print("Number of vertices:", graph.vertices.count())
print("Number of edges:", graph.edges.count())

# จัดกลุ่ม edges ตาม 'src' และ 'dst' พร้อมนับจำนวน, กรองเฉพาะเส้นทางที่นับได้มากกว่า 5, และเรียงลำดับตามจำนวนจากมากไปน้อย
# เพิ่มคอลัมน์สีสำหรับ 'src' และ 'dst' เพื่อการแสดงผล
grouped_edges_df = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") > 5) \
    .orderBy(desc("count")) \
    .withColumn("source_color", lit("#3358FF")) \
    .withColumn("destination_color", lit("#FF3F33"))

# แสดงผล grouped edges ที่ได้
grouped_edges_df.show()

# บันทึกผลลัพธ์ grouped edges ลงในไฟล์ CSV
grouped_edges_df.write.csv("output.csv", mode="overwrite", header=True)
