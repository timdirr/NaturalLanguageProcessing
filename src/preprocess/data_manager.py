from helper import pandas_ndarray_series_to_numpy


class DataManager:
    def __init__(self, lemmatized=False, prune=False, upper=200, lower=10):
        self.lemmatized = lemmatized
        self.prune = prune

        self.upper = upper
        self.lower = lower
        self.train_set = 'dev'  # default dev, can be changed to full

    def __prune_description(self, data):
        data["len"] = data[['description']].apply(lambda x: len(x.values[0].split(' ')), axis=1)
        data = data[(data["len"] > self.lower) & (data["len"] < self.upper)]
        data.drop(columns=["len"], inplace=True)

        summary_marker = ["directed by", "produced by", "starring", "stars as ", "filmed in ", "acted by "]
        pattern = '|'.join(summary_marker)

        data = data[~data['description'].str.contains(pattern, case=False, regex=True)]
        return data

    @property
    def train(self):
        if self.lemmatized:
            X = self._train["lemmatized_description"].apply(lambda row: ' '.join([x for x in row.replace('" "', '').replace('_', '').split() if len(x) > 1]))
            X = X.to_numpy()
            y = pandas_ndarray_series_to_numpy(self._train["genre"])
            return X, y
        else:
            return self._train["description"].to_numpy(), pandas_ndarray_series_to_numpy(self._train["genre"])

    @train.setter
    def train(self, value):
        if self.prune:
            self._train = self.__prune_description(value)
        else:
            self._train = value

    @property
    def dev(self):
        if self.lemmatized:
            X = self._dev["lemmatized_description"].apply(lambda row: ' '.join([x for x in row.replace('" "', '').replace('_', '').split() if len(x) > 1]))
            X = X.to_numpy()
            y = pandas_ndarray_series_to_numpy(self._dev["genre"])
            return X, y
        else:
            return self._dev["description"].to_numpy(), pandas_ndarray_series_to_numpy(self._dev["genre"])

    @dev.setter
    def dev(self, value):
        if self.prune:
            self._dev = self.__prune_description(value)
        else:
            self._dev = value

    @property
    def test(self):
        if self.lemmatized:
            X = self._test["lemmatized_description"].apply(lambda row: ' '.join([x for x in row.replace('" "', '').replace('_', '').split() if len(x) > 1]))
            X = X.to_numpy()
            y = pandas_ndarray_series_to_numpy(self._test["genre"])
            return X, y
        else:
            return self._test["description"].to_numpy(), pandas_ndarray_series_to_numpy(self._test["genre"])

    @test.setter
    def test(self, value):
        if self.prune:
            self._test = self.__prune_description(value)
        else:
            self._test = value

    def __str__(self):
        return f"lemma={self.lemmatized}_prune={self.prune}"
