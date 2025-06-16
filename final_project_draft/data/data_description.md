# Dataset Column Descriptions (train.csv)

This file contains structured booking data. Each row represents a single hotel booking.

## Column List

1. **hotel**  
   Type of hotel (`City Hotel`, `Resort Hotel`).

2. **is_canceled**  
   Whether the booking was canceled (`1` = Yes, `0` = No).  
   _(Target variable)_

3. **lead_time**  
   Number of days between the booking date and the arrival date.

4. **arrival_date_year**  
   Year of arrival.

5. **arrival_date_month**  
   Month of arrival (e.g., July, August).

6. **arrival_date_week_number**  
   Week number of arrival (ISO format).

7. **arrival_date_day_of_month**  
   Day of the month when the guest arrives.

8. **stays_in_weekend_nights**  
   Number of weekend nights (Saturday/Sunday) booked.

9. **stays_in_week_nights**  
   Number of weeknights (Monday to Friday) booked.

10. **adults**  
    Number of adults in the booking.

11. **children**  
    Number of children in the booking. May contain missing values.

12. **babies**  
    Number of babies in the booking.

13. **meal**  
    Type of meal plan (`BB`, `HB`, `FB`, `SC`).

14. **country**  
    Country of origin of the guest (ISO country code).

15. **market_segment**  
    Market segment (e.g., `Online TA`, `Offline TA/TO`, `Direct`).

16. **distribution_channel**  
    Channel through which the booking was made.

17. **is_repeated_guest**  
    Whether the customer is a returning guest (`1` = Yes, `0` = No).

18. **previous_cancellations**  
    Number of previous bookings that were canceled.

19. **previous_bookings_not_canceled**  
    Number of previous bookings that were not canceled.

20. **reserved_room_type**  
    Room type originally reserved.

21. **assigned_room_type**  
    Room type actually assigned to the guest.

22. **booking_changes**  
    Number of changes made to the booking.

23. **deposit_type**  
    Type of deposit (`No Deposit`, `Non Refund`, `Refundable`).

24. **agent**  
    ID of the booking agent (if any).

25. **company**  
    ID of the company making the booking (if any).

26. **days_in_waiting_list**  
    Number of days the booking was on the waiting list before confirmation.

27. **customer_type**  
    Type of customer (`Transient`, `Contract`, `Group`, `Transient-Party`).

28. **adr**  
    Average Daily Rate (price per night).

29. **required_car_parking_spaces**  
    Number of parking spaces required.

30. **total_of_special_requests**  
    Number of special requests made by the customer.

31. **reservation_status**  
    Final reservation status (`Canceled`, `Check-Out`, `No-Show`).

32. **reservation_status_date**  
    Date the last reservation status was set.

---

## 🔒 Confidential Columns (Removed for ML)

33. **name**  
    Full name of the guest. _(Personally Identifiable Information)_

34. **email**  
    Guest's email address. _(Sensitive data)_

35. **phone-number**  
    Guest's phone number. _(Sensitive data)_

36. **credit_card**  
    Credit card number used for the booking. _(Highly sensitive)_

37. **id**  
    Internal user or booking ID. _(Personal identifier)_
