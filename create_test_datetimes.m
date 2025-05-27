% test_datetimes.m script

% Ensure we are dealing with modern datetime objects
try
    % Scalar datetime
    dt_scalar = datetime('2023-10-26');

    % Row vector of datetimes
    dt_row_vector = [datetime('2023-11-01'), datetime('2023-11-15'), datetime('2023-12-25')];

    % Column vector of datetimes
    dt_column_vector = [datetime('2024-01-10'); datetime('2024-02-20'); datetime('2024-03-30')];

    % 2D Matrix of datetimes
    dt_matrix = [datetime('2025-01-01'), datetime('2025-02-01'); datetime('2025-03-01'), datetime('2025-04-01')];

    % Struct containing datetimes
    dt_struct = struct();
    dt_struct.scalar = datetime('2026-07-04');
    dt_struct.array = [datetime('2026-08-01'), datetime('2026-09-15')];
    dt_struct.mixed_datetime_in_struct_array = [struct('d', datetime('2026-10-01')), struct('d', datetime('2026-11-01'))];


    % Cell array containing datetimes
    dt_cell_array = {datetime('2027-05-18'), [datetime('2027-06-01'), datetime('2027-07-01')]};
    dt_cell_array_column = {datetime('2027-08-01'); datetime('2027-09-01')};


    % Datetime with specific time (including milliseconds)
    dt_specific_time = datetime('2023-03-15 14:30:45.678');
    
    % NaT (Not-a-Time)
    dt_nat = NaT; % MATLAB's Not-a-Time constant for datetime arrays
    dt_nat_array = [datetime('2023-01-01'), NaT, datetime('2023-01-03')];

    % Empty datetime array
    dt_empty_array = datetime([]);
    
    % Datetime with timezone (conversion might lose timezone, but good to test storage)
    dt_with_timezone = datetime('2023-10-26 10:00:00', 'TimeZone', 'America/New_York');
    
    % Scalar datetime representing a date far in the past (e.g., year 1500)
    dt_past_scalar = datetime('1500-01-01');
    
    % Scalar datetime representing a date far in the future
    dt_future_scalar = datetime('2500-01-01');

    % Save all variables to a .mat file in v7.3 format
    save('tests/testfile_datetime.mat', ...
        'dt_scalar', 'dt_row_vector', 'dt_column_vector', 'dt_matrix', ...
        'dt_struct', 'dt_cell_array', 'dt_cell_array_column', ...
        'dt_specific_time', 'dt_nat', 'dt_nat_array', 'dt_empty_array', ...
        'dt_with_timezone', 'dt_past_scalar', 'dt_future_scalar', ...
        '-v7.3');

    disp('Successfully created tests/testfile_datetime.mat with datetime objects.');

catch ME
    disp('Error creating test MAT file:');
    disp(ME.message);
    % If running in a context where exit codes matter:
    % exit(1); 
end
