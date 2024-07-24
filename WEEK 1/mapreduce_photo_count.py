from mrjob.job import MRJob

class MapReduce(MRJob):

    def mapper(self, _, line):
        data = line.split(',')
        status_type = data[1].strip()
        if status_type == 'photo':
            yield 'Photo', 1
        if status_type == 'video':
            yield 'Video,', 1
        if status_type == 'link':
            yield 'Link,', 1
        if status_type == 'status':
            yield 'Status', 1

    def reducer(self, key, values):
        yield key, sum(values)

if __name__ == '__main__':
    MapReduce.run()
