# modules related to EDA of new data

import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt

def plot_monthly_class_distribution(input_df, feature_name = 'label', production_start_idx=None, color_palette='husl'):
    """
    Plots the monthly distribution of the specified class in the dataset
    
    Parameters:
    - df: DataFrame containing 'date_captured' and 'label' columns.
    - class_name: The name of the class column to analyze (default is 'label'). can be 'location' 
    
    - production_start_idx: Optional; date to indicate the index where production split starts in time-sorted df.
    """
    df = input_df.copy()
    # Convert 'date_captured' to datetime if not already
    df['date_captured'] = pd.to_datetime(df['date_captured'], errors='coerce')
    
    # Extract year and month from 'date_captured'
    df['year_month'] = df['date_captured'].dt.to_period('M')
    
    # Group by year_month and label, then count occurrences
    monthly_counts = df.groupby(['year_month', feature_name]).size().unstack(fill_value=0)

    # add missing year_month to the monthly_counts DataFrame
    all_periods = pd.period_range(start=df['year_month'].min(), end=df['year_month'].max(), freq='M')
    monthly_counts = monthly_counts.reindex(all_periods, fill_value=0)
    #monthly_counts.index = monthly_counts.index.astype(str)  # Convert PeriodIndex to string for better plotting
    
    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set_palette(color_palette, n_colors=df[feature_name].nunique())
    ax = monthly_counts.plot(kind='bar', stacked=True, figsize=(14, 8))

    # generate another plot for % distribution instead of counts 
    monthly_counts_percentage = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100
    # plot this separately from ax
    ax2 = monthly_counts_percentage.plot(kind='bar', stacked=True, figsize=(14, 8), alpha=0.5)


    
    # If production_start_idx is provided, add a vertical line to the plot
    if production_start_idx:
        # Get the production start date from the index
        production_start_idx = int(production_start_idx)
        # Get the date at the production start index
        production_start_date = df['date_captured'].iloc[production_start_idx]
        production_start_year_month = production_start_date.to_period('M')
        production_start_x = monthly_counts.index.get_loc(production_start_year_month)
        
        

        # split the dataset into modeling and production datasets according to production start idx
        modeling_df = df[:production_start_idx]
        production_df = df[production_start_idx:]

        # get count distribution of labels for modeling and production datasets
        # show distribution of labels in modeling and production datasets
        labels_distribution_df = pd.DataFrame({'label': df['label'].unique()})
        labels_distribution_df.set_index('label', inplace=True)
        # Add modeling and production counts to the labels_distribution_df
        labels_distribution_df['modeling count'] = modeling_df['label'].value_counts().reindex(labels_distribution_df.index, fill_value=0)
        labels_distribution_df['production count'] = production_df['label'].value_counts().reindex(labels_distribution_df.index, fill_value=0)

        current_legend_labels = [t.get_text() for t in ax.legend_.texts]

        # Add modeling and production counts to the legend
        #display(current_legend_labels)
    
        # sort labels distribution df according to current legend labels
        labels_distribution_df = labels_distribution_df.reindex(current_legend_labels, axis=0, fill_value=0)
        display(labels_distribution_df)

        
        
        # Add modeling and production counts to the legend
        legend_labels = [f"{i}: ({labels_distribution_df['modeling count'][i]} - {labels_distribution_df['production count'][i]})" for i in labels_distribution_df.index]

        ax.legend(legend_labels, title='Label (Dev/Prod Count)', loc='upper left')
        #ax.legend(labels_distribution_df, title='Dataset', loc='upper left')
        #display(current_legend_labels)
        plt.axvline(x=production_start_x, color='red', linestyle='--', label='Production Start')

        # add annotations to the vertical line

        image_split_info = f'image file count: {len(modeling_df)} ({len(modeling_df) / len(df) * 100:.2f}%) / {len(production_df)} ({len(production_df) / len(df) * 100:.2f}%)'

        date_span = df['date_captured'].max() - df['date_captured'].min()
        total_days = date_span.days
        production_days = df['date_captured'].max() - production_start_date
        production_days = production_days.days
        modeling_days = total_days - production_days
        #display(f"production days: {production_days}, total days: {total_days}")

        day_split_info = f"days count: {modeling_days} ({modeling_days / total_days * 100:.2f}%) / {production_days} ({(production_days) / total_days * 100:.2f}%)"


        plt.annotate(#f'Dev/Production Split:\nsample index:{production_start_idx}\nimg count : {len(modeling_df)} ({len(modeling_df) / len(df) * 100:.2f}%) / {len(production_df)} ({len(production_df) / len(df) * 100:.2f}%)', 
                     f'DATASET SPLIT TIMING\nsample index:{production_start_idx}\n{image_split_info}\n{day_split_info}', 
                     xy=(production_start_x, monthly_counts.max().max()+2000), 
                     
                     #xytext=(production_start_x + 5, monthly_counts.max().sum() * 0.9),
                     xytext=(production_start_x + 2, (monthly_counts.max().max()+2000) * 0.9),
                     arrowprops=dict(arrowstyle="->", linewidth=1, color='red', alpha = 0.8), 
                     fontsize=10, color='black',
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", lw=1, facecolor='white', alpha=0.8)
        )   
             
    else:
        ax.legend(title='Labels')

    # set x-ticks label to be the actual year and the actual month numbers
    
    #display(monthly_counts.head())
    plt.title(f'Monthly {feature_name} Distribution Over Time')
    plt.xlabel('year_month')
    plt.ylabel('Count')
    #plt.xticks(ticks=range(len(monthly_counts)), labels=monthly_counts['Year-Month'], rotation=90)

  
    # create legend for the feature_name
    #plt.legend(legend)

    plt.tight_layout()
    

    if production_start_idx:
        # show count distributions of labels based on production split 
        sns.set_palette("tab10")
        plt.figure()
        # Create a bar plot for the labels distribution
        sns.barplot(data=labels_distribution_df.reset_index().melt(id_vars='label'), 
                    x='label', y='value', hue='variable')
        # Set the title and labels
        plt.title(f'Image Files Count Distribution for Production Split Idx @ {production_start_idx}')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.legend(title='Dataset')
        # Show the plot
        plt.tight_layout()
        
    
    plt.show()

# plot distribution change in the previous three months for every month in the given df 

def plot_quarterly_class_distribution(input_df, feature_name = 'label', production_start_idx=None, color_palette='husl'):
    """
    Plots the quarterly distribution of the specified class in the dataset
    
    Parameters:
    - df: DataFrame containing 'date_captured' and 'label' columns.
    - class_name: The name of the class column to analyze (default is 'label'). can be 'location' 
    
    - production_start_idx: Optional; date to indicate the index where production split starts in time-sorted df.
    """
    df = input_df.copy()
    # Convert 'date_captured' to datetime if not already
    df['date_captured'] = pd.to_datetime(df['date_captured'], errors='coerce')
    
    # Extract year and month from 'date_captured'
    df['year_month'] = df['date_captured'].dt.to_period('M')
    
    # Group by year_month and label, then count occurrences
    monthly_counts = df.groupby(['year_month', feature_name]).size().unstack(fill_value=0)

    # add missing year_month to the monthly_counts DataFrame
    all_periods = pd.period_range(start=df['year_month'].min(), end=df['year_month'].max(), freq='M')
    monthly_counts = monthly_counts.reindex(all_periods, fill_value=0)
    #monthly_counts.index = monthly_counts.index.astype(str)  # Convert PeriodIndex to string for better plotting

    # from monthly_counts, generate counts for the previous 3 months
    quarterly_counts = monthly_counts.rolling(window=3).sum()
    
    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set_palette(color_palette, n_colors=df[feature_name].nunique())
    ax = quarterly_counts.plot(kind='bar', stacked=True, figsize=(14, 8))

    # generate another plot for % distribution instead of counts 
    quarterly_counts_percentage = quarterly_counts.div(quarterly_counts.sum(axis=1), axis=0) * 100
    # plot this separately from ax
    ax2 = quarterly_counts_percentage.plot(kind='bar', stacked=True, figsize=(14, 8), alpha=0.5)
    
    # If production_start_idx is provided, add a vertical line to the plot
    if production_start_idx:
        # Get the production start date from the index
        production_start_idx = int(production_start_idx)
        # Get the date at the production start index
        production_start_date = df['date_captured'].iloc[production_start_idx]
        production_start_year_month = production_start_date.to_period('M')
        production_start_x = monthly_counts.index.get_loc(production_start_year_month)
        
        

        # split the dataset into modeling and production datasets according to production start idx
        modeling_df = df[:production_start_idx]
        production_df = df[production_start_idx:]

        # get count distribution of labels for modeling and production datasets
        # show distribution of labels in modeling and production datasets
        labels_distribution_df = pd.DataFrame({'label': df['label'].unique()})
        labels_distribution_df.set_index('label', inplace=True)
        # Add modeling and production counts to the labels_distribution_df
        labels_distribution_df['modeling count'] = modeling_df['label'].value_counts().reindex(labels_distribution_df.index, fill_value=0)
        labels_distribution_df['production count'] = production_df['label'].value_counts().reindex(labels_distribution_df.index, fill_value=0)

        current_legend_labels = [t.get_text() for t in ax.legend_.texts]

        # Add modeling and production counts to the legend
        #display(current_legend_labels)
    
        # sort labels distribution df according to current legend labels
        labels_distribution_df = labels_distribution_df.reindex(current_legend_labels, axis=0, fill_value=0)
        display(labels_distribution_df)

        
        
        # Add modeling and production counts to the legend
        legend_labels = [f"{i}: ({labels_distribution_df['modeling count'][i]} - {labels_distribution_df['production count'][i]})" for i in labels_distribution_df.index]

        ax.legend(legend_labels, title='Label (Dev/Prod Count)', loc='upper left')
        #ax.legend(labels_distribution_df, title='Dataset', loc='upper left')
        #display(current_legend_labels)
        plt.axvline(x=production_start_x, color='red', linestyle='--', label='Production Start')

        # add annotations to the vertical line

        image_split_info = f'image file count: {len(modeling_df)} ({len(modeling_df) / len(df) * 100:.2f}%) / {len(production_df)} ({len(production_df) / len(df) * 100:.2f}%)'

        date_span = df['date_captured'].max() - df['date_captured'].min()
        total_days = date_span.days
        production_days = df['date_captured'].max() - production_start_date
        production_days = production_days.days
        modeling_days = total_days - production_days
        #display(f"production days: {production_days}, total days: {total_days}")

        day_split_info = f"days count: {modeling_days} ({modeling_days / total_days * 100:.2f}%) / {production_days} ({(production_days) / total_days * 100:.2f}%)"


        plt.annotate(#f'Dev/Production Split:\nsample index:{production_start_idx}\nimg count : {len(modeling_df)} ({len(modeling_df) / len(df) * 100:.2f}%) / {len(production_df)} ({len(production_df) / len(df) * 100:.2f}%)', 
                     f'DATASET SPLIT TIMING\nsample index:{production_start_idx}\n{image_split_info}\n{day_split_info}', 
                     xy=(production_start_x, monthly_counts.max().max()+2000), 
                     
                     #xytext=(production_start_x + 5, monthly_counts.max().sum() * 0.9),
                     xytext=(production_start_x + 2, (monthly_counts.max().max()+2000) * 0.9),
                     arrowprops=dict(arrowstyle="->", linewidth=1, color='red', alpha = 0.8), 
                     fontsize=10, color='black',
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", lw=1, facecolor='white', alpha=0.8)
        )   
             
    else:
        ax.legend(title='Labels')

    # set x-ticks label to be the actual year and the actual month numbers
    
    #display(monthly_counts.head())
    plt.title(f'Moving Quarterly {feature_name} Distribution Over Time')
    plt.xlabel('year_month')
    plt.ylabel('Count')
    #plt.xticks(ticks=range(len(monthly_counts)), labels=monthly_counts['Year-Month'], rotation=90)

  
    # create legend for the feature_name
    #plt.legend(legend)

    plt.tight_layout()
    

    if production_start_idx:
        # show count distributions of labels based on production split 
        sns.set_palette("tab10")
        plt.figure()
        # Create a bar plot for the labels distribution
        sns.barplot(data=labels_distribution_df.reset_index().melt(id_vars='label'), 
                    x='label', y='value', hue='variable')
        # Set the title and labels
        plt.title(f'Image Files Count Distribution for Production Split Idx @ {production_start_idx}')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.legend(title='Dataset')
        # Show the plot
        plt.tight_layout()
        
    
    plt.show()