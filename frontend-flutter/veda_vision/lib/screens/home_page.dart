import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import 'flower_tab.dart';
import 'leaf_tab.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [AppColors.bgStart, AppColors.bgEnd],
            ),
          ),
          child: SafeArea(
            child: Column(
              children: [
                const Padding(
                  padding: EdgeInsets.fromLTRB(24, 20, 24, 4),
                  child: Text(
                    '🌿 Ayurveda Plant Recognition',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: AppColors.titleColor,
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 24),
                  decoration: BoxDecoration(
                    color: AppColors.cardBg,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: TabBar(
                    indicator: BoxDecoration(
                      color: AppColors.buttonBg,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    indicatorSize: TabBarIndicatorSize.tab,
                    labelColor: Colors.white,
                    unselectedLabelColor: AppColors.textPrimary,
                    labelStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
                    tabs: const [
                      Tab(text: '🌸 Flower'),
                      Tab(text: '🍃 Leaves'),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                const Expanded(
                  child: TabBarView(
                    children: [
                      FlowerTab(),
                      LeafTab(),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}