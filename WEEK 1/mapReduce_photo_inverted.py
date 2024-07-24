from mrjob.job import MRJob

class MapReduce(MRJob):
    def mapper(self, _, line):
        if 'status_id' not in line:

            data = line.split(",")

            status_type = data[1].strip()

            num_reactions = data[3].strip()

            yield status_type, num_reactions


    def reducer(self, key, values):

        list = []
        for react in values:
            list.append(react)

        yield key, list

if __name__ == '__main__':
    MapReduce.run()

