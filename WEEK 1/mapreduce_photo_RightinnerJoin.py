from mrjob.job import MRJob
from mrjob.step import MRStep

class MapReduceInnerJoin(MRJob):

    def mapper(self, _, line):
        data = line.split(',')
        fbId = data[1]
        yield fbId, data

    def reducer(self, key, values):
        fb2 = []
        fb3 = []

        for v in values:
            if v[0] == 'FB2':
                fb2.append(v)
            if v[0] == 'FB3':
                fb3.append(v)

        for i in fb3:
            if len(fb2) > 0:
                for j in fb2:
                    yield None, (j+i)
        
            if len(fb2) == 0:
                yield None, i
    def steps(self):
        return [MRStep(mapper=self.mapper, reducer=self.reducer)]

if __name__ == '__main__':
    MapReduceInnerJoin.run()
