from mrjob.job import MRJob
from mrjob.step import MRStep
import heapq

class MapReduce(MRJob):
    def mapper(self, _, line):
        if 'status_id' not in line:
            data = line.split(",")
            date = data[2].strip()
            year = date.split(' ')[0].split('/')[2]
            status_type = data[1].strip()
            if status_type in ['photo', 'link', 'video', 'status']:
                yield (year, status_type), 1

    def reducer_count(self, key, values):
        yield key[0], (sum(values), key[1])

    def reducer_max(self, key, values):
        top_n = heapq.nlargest(2, values)
        result = []
        for count, status_type in top_n:
            result.append([count, status_type])
        yield key, result

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer_count),
            MRStep(reducer=self.reducer_max)
        ]

if __name__ == '__main__':
    MapReduce.run()