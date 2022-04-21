from mysklearn import myutils
import copy
import csv
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows = len(self.data)
        cols = len(self.column_names)
        return rows, cols 

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []

        try:
            if type(col_identifier) == int:
                col_index = col_identifier
                for row in self.data:
                    value = row[col_index]
                    if value != "NA":
                        col.append(value)
            elif type(col_identifier) == str:
                col_index = self.column_names.index(col_identifier)
                for row in self.data:
                    value = row[col_index]
                    if value != "NA":
                        col.append(value)
        except ValueError:
            print("Value Error: column identifier is not a string or int")

        return col 

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for item in row:
                index = self.data.index(row)
                item_index = self.data[index].index(item)
                try:
                    temp = float(self.data[index][item_index])
                    self.data[index][item_index] = temp
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for row in reversed(row_indexes_to_drop):
            self.data.pop(row)
  
    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, 'r') as file:
            csvreader = csv.reader(file)
            self.column_names = next(csvreader)
            for row in csvreader:
                self.data.append(row)

        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile)  
            csvwriter.writerow(self.column_names) 
            # writing the data rows 
            csvwriter.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        
        unique = []
        duplicate_ids = []
        keyed_columns = []
        for row in self.data:
            temp = []
            for col in key_column_names:
                col_to_search = self.column_names.index(col)
                temp.append(row[col_to_search])
            keyed_columns.append(temp)

        row_num = 0
        for row in keyed_columns:
            if row not in unique:
                unique.append(row)
            else:
                duplicate_ids.append(row_num)
            row_num += 1
        
        return duplicate_ids 

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        for row in range(len(self.data)):
            for item in row:
                if '' == item:
                    del self.data[row]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        total = 0
        count = 0
        col_index = self.column_names.index(col_name)
        for row in self.data:
            if type(row[col_index]) == float:
                count += 1
                total += row[col_index]
        avg = total / count
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        summary_header = ["attribute", "min", "max", "mid", "avg", "median"]
        summary_stats = []

        if len(self.data) == 0:
            return MyPyTable([], [])
        for col in col_names:
            min = 0.0
            max = 0.0
            mid = 0
            avg = 0
            median = 0
            total = 0
            tmp = []

            for row in self.data:
                if row[self.column_names.index(col)] < min:
                    min = row[self.column_names.index(col)]
                if row[self.column_names.index(col)] > max:
                    max = row[self.column_names.index(col)]
                total += row[self.column_names.index(col)]
                tmp.append(row[self.column_names.index(col)])
            avg = total / len(self.data)
            mid = (min + max) / 2
            tmp.sort()
            # if the length is even it finds the middle of the two values
            if len(self.data) % 2 == 0:
                middle_lower = tmp[int(len(tmp) / 2) - 1]
                middle_higher = tmp[int(len(tmp) / 2)]
                median =  (middle_lower + middle_higher) / 2
            else:
                median = middle_higher = tmp[int(len(tmp) / 2)]
            tmp = [col, min, max, mid, avg, median]
            summary_stats.append(tmp)

        return MyPyTable(summary_header, summary_stats)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        navigate_loop = True
        index = 0
        key_index_out_of_order = 1
        indexes_match = True # check if key_column names are in the same index for each table
        matched_at_muplitple_indexes = 0
        new_column_names = []
        new_row = []
        final_table = MyPyTable() 
        key_iterator = 0
        self_key_indexes = []
        other_key_indexes = []
        column_index = 0

        # get the column indexes
        for item in self.column_names:
            for key in key_column_names:
                if (key == item):
                    self_key_indexes.append(column_index)
            column_index += 1
        # reset before moving to next table
        column_index = 0 

        for item in other_table.column_names:
            for key in key_column_names:
                if (key == item):
                    other_key_indexes.append(column_index)
            column_index += 1

        # combine table headers
        for item in self.column_names:
            new_column_names.append(item)
        for item in other_table.column_names:
            new_column_names.append(item)
            
        # duplicate column names
        for item in new_column_names:
            if new_column_names.count(item) > 1:
                new_column_names.pop(key_iterator)
            key_iterator += 1

        # column names match indexes
        # keep index in the right range
        if (len(self_key_indexes) > 0 and len(other_key_indexes)): 
            if (self.column_names[self_key_indexes[0]] != other_table.column_names[other_key_indexes[0]]):
                indexes_match = False
        # add new headers to new table
        final_table.column_names = new_column_names
        # reset key_iterator
        key_iterator = 0

        for self_row in self.data:
            for other_row in other_table.data:
                # checks for matches matches
                for index in range(len(self_key_indexes)): 
                    if (indexes_match == True):
                        if (self_row[self_key_indexes[index]] == other_row[other_key_indexes[index]]):
                            matched_at_muplitple_indexes += 1
                        if (matched_at_muplitple_indexes < len(self_key_indexes)):
                            navigate_loop = False
                        else:
                            navigate_loop = True
                    else:
                        if (self_row[self_key_indexes[index]] == other_row[other_key_indexes[key_index_out_of_order]]):
                            matched_at_muplitple_indexes += 1
                        # reset for next loop
                        key_index_out_of_order = 0 
                        if (matched_at_muplitple_indexes < len(self_key_indexes)):
                            navigate_loop = False
                        else:
                            navigate_loop = True
                # reset for next loop
                key_index_out_of_order = 1 
                if (navigate_loop == True):
                    for item in self_row:
                        new_row = new_row + [item]
                    for index in range(len(other_row)):
                        # columns that aren't keys to new_row
                        if (other_key_indexes.count(index) < 1): 
                            new_row = new_row + [other_row[index]]
                        index += 1
                matched_at_muplitple_indexes = 0
                # turn True in case not a match
                navigate_loop = True 
                # row has data
                if (len(new_row) > 0): 
                    final_table.data.append(new_row)
                new_row = []         
        key_iterator += 1

        return final_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        navigate_loop = True
        row_added = False
        index = 0
        key_index_out_of_order = 1 
        indexes_match = True # check if key_column names are in the same index for each table
        matched_at_muplitple_indexes = 0 
        new_column_names = []
        new_row = []
        final_table = MyPyTable() 
        key_iterator = 0 
        self_key_indexes = [] 
        other_key_indexes = [] 
        column_index = 0
        
        # get the column indexes
        for item in self.column_names:
            for key in key_column_names:
                if (key == item):
                    self_key_indexes.append(column_index)
            column_index += 1
        # reset before moving to next table
        column_index = 0 

        for item in other_table.column_names:
            for key in key_column_names:
                if (key == item):
                    other_key_indexes.append(column_index)
            column_index += 1 

        # combine table headers
        for item in self.column_names:
            new_column_names.append(item)
        for item in other_table.column_names:
            if key_column_names.count(item) < 1:
                new_column_names.append(item)
        # column names match indexes
        # keep index in the right range
        if (len(self_key_indexes) > 0 and len(other_key_indexes)): 
            if (self.column_names[self_key_indexes[0]] != other_table.column_names[other_key_indexes[0]]):
                indexes_match = False
        # add new headers to new table
        final_table.column_names = new_column_names
        # reset key_iterator
        key_iterator = 0

        for self_row in self.data:
            for other_row in other_table.data:
                # checks for matches matches
                for index in range(len(self_key_indexes)): 
                    if (indexes_match == True):
                        if (self_row[self_key_indexes[index]] == other_row[other_key_indexes[index]]):
                            matched_at_muplitple_indexes += 1
                        if (matched_at_muplitple_indexes < len(self_key_indexes)):
                            navigate_loop = False
                        else:
                            navigate_loop = True
                    else:
                        if (self_row[self_key_indexes[index]] == other_row[other_key_indexes[key_index_out_of_order]]):
                            matched_at_muplitple_indexes += 1
                        # reset for next loop
                        key_index_out_of_order = 0 
                        if (matched_at_muplitple_indexes < len(self_key_indexes)):
                            navigate_loop = False
                        else:
                            navigate_loop = True
                # reset for next loop
                key_index_out_of_order = 1
                if (navigate_loop == True):
                    for item in self_row:
                        new_row = new_row + [item]
                    for index in range(len(other_row)):
                        # columns that aren't keys to new_row
                        if (other_key_indexes.count(index) < 1): 
                            new_row = new_row + [other_row[index]]
                        index += 1
                matched_at_muplitple_indexes = 0
                # turn True in case not a match
                navigate_loop = True 
                # does list have data
                if (len(new_row) > 0): 
                    final_table.data.append(new_row)
                    row_added = True
                new_row = []
            if (row_added == True):
                # reset for next loop and continue to next row
                row_added = False
            # outer join
            else: 
                new_row = self_row
                for index in range(len(other_row)):
                    # adds columns that aren't keys
                    if (other_key_indexes.count(index) < 1): 
                        new_row = new_row + ["NA"]
                    index += 1
                final_table.data.append(new_row)
                new_row = []
        key_iterator += 1

        # outer join on unmatched rows in table 2
        matched_at_muplitple_indexes = 0
        for other_row in other_table.data:
            for row in final_table.data:
                # checks for matches matches
                for index in range(len(self_key_indexes)):
                    if (indexes_match == True):
                        if (row[self_key_indexes[index]] == other_row[other_key_indexes[index]]):
                            matched_at_muplitple_indexes += 1
                        if (matched_at_muplitple_indexes < len(self_key_indexes)):
                            navigate_loop = False
                        else:
                            navigate_loop = True
                            break
                    else:
                        if (row[self_key_indexes[index]] == other_row[other_key_indexes[key_index_out_of_order]]):
                            matched_at_muplitple_indexes += 1
                        # reset for next loop
                        key_index_out_of_order = 0 
                        if (matched_at_muplitple_indexes < len(self_key_indexes)):
                            navigate_loop = False
                        else:
                            navigate_loop = True
                            break
                # reset for next loop
                key_index_out_of_order = 1
                matched_at_muplitple_indexes = 0
                # contains a match
                if (navigate_loop == True): 
                    break
            
            # no match found
            if (navigate_loop == False): 
                new_row = []
                key_index_out_of_order = 1
                index = 0
                index_key_iterator = 0
                for index in range(len(self.column_names)):
                    # columns that aren't keys to new_row
                    if (self_key_indexes.count(index) < 1):
                        new_row = new_row + ["NA"]
                    # item is a key
                    else: 
                        if (indexes_match == True):
                            new_row = new_row + [other_row[other_key_indexes[index_key_iterator]]]
                            index_key_iterator += 1
                        # indexe doesn't have a match
                        else: 
                            new_row = new_row + [other_row[other_key_indexes[key_index_out_of_order]]]
                            key_index_out_of_order  = key_index_out_of_order - 1
                for index in range(len(other_row)):
                    # columns that aren't keys are added
                    if (other_key_indexes.count(index) < 1): 
                        new_row = new_row + [other_row[index]]
                    index += 1
                final_table.data.append(new_row)
                new_row = []
            # reset for next loop
            navigate_loop = False
            
        return final_table
