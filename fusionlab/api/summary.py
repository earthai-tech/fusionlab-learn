# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
The `summary` module offers a comprehensive suite of tools for
generating and formatting summaries and reports, ensuring clear and consistent
 presentation of analytical results.
"""

import copy

from .bunch import FlexDict
from .util import to_snake_case, get_table_size 
from .util import  to_camel_case  
from .util import beautify_dict
TW = get_table_size()

class ResultSummary(FlexDict):
    """
    Initializes a ResultSummary object which can store, format, and display
    results in a structured format. The class allows for optional customization
    of the display settings such as padding of keys and maximum character limits
    for value display.
    
    Parameters
    ----------
    name : str, optional
        The name of the result set, which will be displayed as the title
        of the output. Defaults to "Result".
    pad_keys : str, optional
        If set to "auto", keys in the result dictionary will be left-padded
        to align with the longest key, enhancing readability.
    max_char : int, optional
        The maximum number of characters that a value can have before being
        truncated. Defaults to ``100``.
    flatten_nested_dicts : bool, optional
        Determines whether nested dictionaries within the results should be 
        displayed in a flattened, one-line format.
        When set to ``True``, nested dictionaries are presented as a compact single
        line, which might be useful for brief overviews or when space is limited.
        If set to ``False``, nested dictionaries are displayed with full 
        indentation and key alignment, which improves readability for complex 
        structures. Defaults to ``True``.
    mute_note: bool, default=False 
       Skip displaying the note after result formatage. 
       
    Examples
    --------
    >>> from fusionlab.api.summary import ResultSummary
    >>> summary = ResultSummary(name="Data Check", pad_keys="auto", max_char=50)
    >>> results = {
        'long_string_data': "This is a very long string that needs truncation.",
        'data_counts': {'A': 20, 'B': 15}
    }
    >>> summary.add_results(results)
    >>> print(summary)
    DataCheck(
      {
        long_string_data          : "This is a very long string that needs trunc..."
        data_counts               : {'A': 20, 'B': 15}
      }
    )
    """
    def __init__(self, name=None, pad_keys=None, max_char=None,
                 flatten_nested_dicts =True, mute_note=False, 
                 **kwargs):

        super().__init__(**kwargs)
        self.name = name or "Result"
        self.pad_keys = pad_keys
        self.max_char = max_char or get_table_size()
        self.flatten_nested_dicts = flatten_nested_dicts 
        self.mute_note=mute_note
        self.results = {}
        
    def add_results(self, results):
        """
        Adds results to the summary and dynamically creates attributes for each
        key in the results dictionary, converting keys to snake_case for attribute
        access.

        Parameters
        ----------
        results : dict
            A dictionary containing the results to add to the summary. Keys should
            be strings and will be converted to snake_case as object attributes.

        Raises
        ------
        TypeError
            If the results parameter is not a dictionary.

        Examples
        --------
        >>> summary = ResultSummary()
        >>> summary.add_results({'Missing Data': {'A': 20}})
        >>> print(summary.missing_data)
        {'A': 20}
        """
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary")
    
        # Deep copy to ensure that changes to input dictionary 
        # do not affect internal state
        self.results = copy.deepcopy(results)
    
        # Apply snake_case to dictionary keys and set attributes
        for name in list(self.results.keys()):
            snake_name = to_snake_case(name)
            setattr(self, snake_name, self.results[name])
            
        return self 

    def __str__(self):
        """
        Return a formatted string representation of the results dictionary.
        """
        _name = to_camel_case(self.name)
        result_title = _name + '(\n  {\n'
        formatted_results = []
        
        # Determine key padding if auto pad_keys is specified
        if self.pad_keys == "auto":
            max_key_length = max(len(key) for key in self.results.keys())
            key_padding = max_key_length
        else:
            key_padding = 0  # No padding by default
    
        # Construct the formatted result string
        for key, value in self.results.items():
            if self.pad_keys == "auto":
                formatted_key = key.ljust(key_padding)
            else:
                formatted_key = key
            if isinstance(value, dict):
                if self.flatten_nested_dicts: 
                    value_str= str(value)
                else: 
                    value_str = beautify_dict(
                        value, key=f"       {formatted_key}",
                        max_char= self.max_char
                        ) 
                    formatted_results.append(value_str +',') 
                    continue 
            else:
                value_str = str(value)
            
            # Truncate values if necessary
            if len(value_str) > self.max_char:
                value_str = value_str[:self.max_char] + "..."
            
            formatted_results.append(f"       {formatted_key} : {value_str}")
    
        result_str = '\n'.join(formatted_results) + "\n\n  }\n)"
        entries_summary = f"[ {len(self.results)} entries ]"
        result_str += f"\n\n{entries_summary}"
        # If ellipsis (...) is present in the formatted result, it indicates 
        # that some data has been truncated
        # for brevity. For the complete dictionary result, please access the 
        # corresponding attribute."
        # note = ( "\n\n[ Note: Data may be truncated. For the complete dictionary"
        #         " data, access the corresponding 'results' attribute ]"
        #         ) if "..." in result_str else ''
        note = (
            "\n\n[ Note: Output may be truncated. To access the complete data,"
            f" use the `results` attribute of the {_name} object:`<obj>.results`. ]"
            ) if "..." in result_str else ''

        if self.mute_note: 
            note =''
        return f"{result_title}\n{result_str}{note}"

    def __repr__(self):
        """
        Return a developer-friendly representation of the ResultSummary.
        """
        name =to_camel_case(self.name)
        return ( f"<{name} with {len(self.results)} entries."
                " Use print() to see detailed contents.>") if self.results else ( 
                    f"<Empty {name}>")
