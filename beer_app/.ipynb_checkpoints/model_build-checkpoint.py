{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "from surprise import KNNWithMeans\n",
    "from surprise import Dataset\n",
    "from surprise.model_selection import GridSearchCV\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import surprise as surp\n",
    "from surprise import BaselineOnly\n",
    "from surprise import Dataset\n",
    "from surprise import Reader\n",
    "from surprise.model_selection import cross_validate\n",
    "from surprise import KNNWithZScore\n",
    "from surprise import KNNBasic\n",
    "\n",
    "import pickle as pk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "beer = pd.read_csv(\"https://www.dropbox.com/s/iqcsdech6e0b9xg/beer_reviews.csv?dl=1\")\n",
    "beer_sub = beer[['review_profilename', 'beer_beerid', 'review_overall']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  review_profilename  beer_beerid  review_overall\n",
      "0            stcules        47986             1.5\n",
      "1            stcules        48213             3.0\n",
      "2            stcules        48215             3.0\n",
      "3            stcules        47969             3.0\n",
      "4     johnmichaelsen        64883             4.0\n"
     ]
    }
   ],
   "source": [
    "beer_counts = beer_sub[['review_profilename']].groupby(['review_profilename']).size()\n",
    "beer_counts = beer_counts[beer_counts > 1000]\n",
    "print(beer_sub[beer_sub['review_profilename'].isin(beer_counts.index)].head(5))\n",
    "reader = Reader(rating_scale=(1, 5))\n",
    "beer_small = beer_sub[beer_sub['review_profilename'].isin(beer_counts.index)]\n",
    "beer_full_small = beer[beer_sub['review_profilename'].isin(beer_counts.index)]\n",
    "reader = Reader(rating_scale=(1, 5))\n",
    "beer_surp = surp.dataset.Dataset.load_from_df(beer_small, reader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Estimating biases using als...\n",
      "Computing the pearson_baseline similarity matrix...\n",
      "Done computing similarity matrix.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<surprise.prediction_algorithms.knns.KNNWithMeans at 0x7f421d08b860>"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainsetfull = beer_surp.build_full_trainset()\n",
    "sim_options = {'name': 'pearson_baseline', 'user_based': False}\n",
    "algo = KNNWithMeans(sim_options = sim_options)\n",
    "algo.fit(trainsetfull)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "A = map(lambda x: pd.DataFrame({'inner_id': x, \\\n",
    "                                'raw_id': algo.trainset.to_raw_iid(x), \\\n",
    "                                'beer_name': beer_mappings.beer_name.iloc[x], \\\n",
    "                                'neighbors': algo.get_neighbors(x, k = 10), \\\n",
    "                                'neighbors_rawid': list(map(lambda xx: algo.trainset.to_raw_iid(xx), algo.get_neighbors(x, k = 10))), \\\n",
    "                                'neighbor_names': list(map(lambda y: beer_mappings.beer_name.iloc[y], algo.get_neighbors(x, k = 10)))}), trainsetfull.all_items())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('knnmodel.pickle', 'wb') as handle:\n",
    "    pk.dump(algo, handle, protocol=pk.HIGHEST_PROTOCOL)\n",
    "with open('beer_small.pickle', 'wb') as handle:\n",
    "    pk.dump(beer_small, handle, protocol=pk.HIGHEST_PROTOCOL)    \n",
    "with open('beer_full_small.pickle', 'wb') as handle:\n",
    "    pk.dump(beer_full_small, handle, protocol=pk.HIGHEST_PROTOCOL)        \n",
    "with open('neighbour_algo.pickle', 'wb') as handle:\n",
    "    pk.dump(algo.get_neighbors, handle, protocol=pk.HIGHEST_PROTOCOL)            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('neighbour_algo.pickle', 'wb') as handle:\n",
    "    pk.dump(algo.get_neighbors, handle, protocol=pk.HIGHEST_PROTOCOL)            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
