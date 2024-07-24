from pyspark.sql import SparkSession

# สร้าง SparkSession
spark = SparkSession.builder.getOrCreate()

# ข้อมูลที่เป็นตัวอักษร
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# สร้าง RDD และแบ่งข้อมูลเป็น 4 partitions
rdd = spark.sparkContext.parallelize(alphabet, 4)
print('Number of partitions: ' + str(rdd.getNumPartitions()))

# โหลดไฟล์ CSV เป็น RDD และแบ่งเป็น 5 partitions
rdd2 = spark.sparkContext.textFile('fb_live_thailand.csv', 5)
print('Number of partitions: ' + str(rdd2.getNumPartitions()))

# โหลดไฟล์ CSV เป็น RDD โดยใช้ wholeTextFile และแบ่งเป็น 5 partitions
rdd3 = spark.sparkContext.wholeTextFile('fb_live_thailand.csv', 5)
print('Number of partitions: ' + str(rdd3.getNumPartitions()))

# ใช้ flatMap กับข้อมูลตัวอักษร (ถึงแม้ว่าจะไม่จำเป็นในกรณีนี้เพราะข้อมูลไม่มีคอมม่า)
flatmap = rdd.flatMap(lambda x: x.split(','))

# สร้างคู่ข้อมูล (key, value) โดย key เป็นตัวอักษรและ value เป็น 1
pair = flatmap.map(lambda x: (x, 1))

# ดึงข้อมูลตัวอย่าง 200 คู่
take_pair = pair.take(200)

# กรองและพิมพ์คู่ข้อมูลที่ key เป็น 'photo' หรือ 'video'
for f in take_pair:
    if str(f[0]) == 'photo' or str(f[0]) == 'video':
        print(str(f[0]), str(f[1]))
