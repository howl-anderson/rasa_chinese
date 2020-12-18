class BatchingIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, data, *args, **kwargs):
        mini_batch = []

        for i in data:
            mini_batch.append(i)

            if len(mini_batch) == self.batch_size:
                yield mini_batch

                # reset
                mini_batch = []

        if mini_batch:  # if there are some residual data
            yield mini_batch


if __name__ == "__main__":
    data = range(19)

    bi = BatchingIterator(5)
    for i in bi(data):
        print(i)
