from mrjob.job import MRJob

class MapReduce(MRJob):
    def mapper(self, _, line):

        if 'status_id' not in line:

            data = line.split(",")

            date = data[2].strip()

            year = date.split(' ')[0].split('/')[2]

            status_type = data[1].strip()


        
            status_type = data[1].strip().lower()

            num_reactions = data[3].strip()
            
            if year == '2018' and int(num_reactions) > 2000:
                yield status_type, num_reactions

    # def reducer(self, key, values):
    #     yield key, sum(values)

if __name__ == '__main__':
    MapReduce.run()

