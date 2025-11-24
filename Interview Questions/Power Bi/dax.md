# DAX Interview Questions (Beginner → Advanced)

## Beginner-Level DAX Questions

1. What is DAX?

2. Why is DAX used in Power BI?

3. What is the difference between Calculated Columns and Measures?

4. What is a Calculated Table?

5. What is a Row Context?

6. What is a Filter Context?

7. What is the difference between Row Context and Filter Context?

8. What is Evaluation Context in DAX?

9. What is the difference between SUM and SUMX?

10. What does the CALCULATE() function do?

11. What does the FILTER() function do?

12. What is the difference between COUNT, COUNTA, COUNTX, COUNTROWS?

13. What is RELATED() used for?

14. What is RELATEDTABLE() used for?

15. What is EARLIER() used for?

16. What is ALL() used for?

17. What is ALLEXCEPT()?

18. What is DISTINCT()?

19. What is VALUES()?

20. What is DIVIDE() and why is it used instead of /?

21. What are Time Intelligence functions?

22. What is SAMEPERIODLASTYEAR()?

23. What is TOTALYTD(), TOTALMTD(), TOTALQTD()?

24. What is MIN(), MAX(), AVERAGE() in DAX?

25. What is the IF() function?

26. What is SWITCH() function and when to use it?

## Intermediate-Level DAX Questions

27. What is a virtual table in DAX?

28. Difference between SUMMARIZE() and SUMMARIZECOLUMNS().

29. What is CALCULATETABLE()?

30. How does the ALLSELECTED() function work?

31. What is REMOVEFILTERS()?

32. What is KEEPFILTERS()?

33. What is CROSSFILTER()?

34. What is USERELATIONSHIP()?

35. How do you activate an inactive relationship using DAX?

36. What is TREATAS() and when is it used?

37. What is VAR keyword in DAX and why is it useful?

38. What is SELECTEDVALUE()?

39. Difference between SELECTEDVALUE() and VALUES().

40. What is HASONEVALUE()?

41. What is HASONEFILTER()?

42. What is ISFILTERED()?

43. What is CALCULATE() + FILTER() combination used for?

44. What is RANKX()? How does it work?

45. Difference between ALL(), ALLSELECTED(), ALLEXCEPT().

46. What is TOPN() function?

47. What is ADDCOLUMNS()?

48. What is GENERATE() function?

49. What is GENERATEALL()?

50. What is INTERSECT(), UNION(), EXCEPT() in DAX?

51. What is GROUPBY() vs SUMMARIZE()?

52. What is NATURALINNERJOIN() and NATURALLEFTOUTERJOIN()?

53. What is an Iterator Function?

54. Difference between SUM() and SUMX(), MAX() and MAXX(), etc.

55. When should we prefer measures over calculated columns?

## Advanced DAX Questions

56. What is context transition in DAX?

57. What is the difference between Context Transition and Row Context?

58. What is expanded tables concept in DAX?

59. Explain how CALCULATE() performs context transition.

60. Explain how FILTER() creates a row context inside CALCULATE().

61. What is bidirectional filtering and how does DAX interact with it?

62. What is the VertiPaq storage engine?

63. How does DAX affect performance in VertiPaq?

64. What is the difference between Storage Engine (SE) and Formula Engine (FE)?

65. What are the main performance bottlenecks in DAX queries?

66. What is the difference between VAR and RETURN in DAX?

67. How do you debug or optimize a DAX measure?

68. What is an "Iterator with Row Context + Context Transition" issue?

69. How does the CALCULATE() function override filter context?

70. What is the purpose of CALCULATETABLE() in advanced modeling?

71. How does TREATAS() manipulate filter context internally?

72. What is the use of SUMMARIZECOLUMNS() in performance optimization?

73. How does KEEPFILTERS() change the behavior of CALCULATE()?

74. How does CROSSFILTER() change relationship directions?

75. What is a semi-additive measure (e.g., end-of-month balance)?

76. How do you write Opening Balance and Closing Balance using DAX?

77. How do you calculate Running Total using DAX?

78. How to calculate Moving Average in DAX?

79. What is the difference between ALL() and REMOVEFILTERS() in advanced scenarios?

80. Explain the internal behavior of SELECTEDVALUE() when no selection is made.

81. What is a disconnected table and how is it used with DAX?

82. How to implement dynamic titles using DAX?

83. How to dynamically switch measures using DAX (AMR technique)?

84. How does DAX handle circular dependencies?

85. What are parent-child hierarchies and PATH(), PATHITEM() functions?

86. What is the RLS (Row-Level Security) filter logic in DAX?

87. Write a measure that calculates % Contribution of each row to the total.

88. Write a measure to calculate YoY, QoQ, MoM growth.

89. Write a measure for dynamic ranking with filters applied.

90. Create a measure that returns the last non-blank value (using LASTNONBLANK).

91. How to calculate KPI thresholds using DAX?

92. What is the difference between CALCULATE() and CALCULATETABLE()?

93. What are Boolean filter expressions in DAX?

94. How does the FILTER function apply lazy evaluation?

95. What is a row-by-row iterator vs set-based logic in DAX?

96. What is the performance difference between SUMX() vs SUM()?

97. How to use variables to reduce repeated calculations?

98. How to handle complex KPI calculations using SWITCH + TRUE()?

99. What are composite models and how does DAX behave with DirectQuery?

100. What are advanced time intelligence patterns for non-standard calendars?

